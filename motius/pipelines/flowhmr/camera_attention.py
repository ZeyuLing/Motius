"""Measured selection of installed exact SDPA kernels; no attention sparsification."""
import time
import torch
import torch.nn.functional as F


class CameraAttention:
    def __init__(self, mode='auto'):
        if mode not in ('auto', 'torch', 'cudnn'):
            raise ValueError('Camera attention mode must be auto, torch or cudnn')
        self.mode = mode
        self.selected = None
        self.timings = {}
        self.fallback_reasons = []

    def _run(self, name, q, k, v):
        if name == 'torch':
            return F.scaled_dot_product_attention(q, k, v)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)

    def _measure(self, name, q, k, v):
        self._run(name, q, k, v)  # Exclude kernel/plan warmup from selection.
        torch.cuda.synchronize(q.device)
        start = time.perf_counter()
        for _ in range(3):
            self._run(name, q, k, v)
        torch.cuda.synchronize(q.device)
        return (time.perf_counter()-start)/3

    def __call__(self, q, k, v):
        if self.selected is None:
            if self.mode != 'auto':
                self.selected = self.mode
            elif not q.is_cuda or k.shape[-2] < 8192:
                self.selected = 'torch'
            else:
                self.timings['torch'] = self._measure('torch', q, k, v)
                try:
                    self.timings['cudnn'] = self._measure('cudnn', q, k, v)
                except RuntimeError as error:
                    self.fallback_reasons.append(str(error))
                    torch.cuda.empty_cache()
                self.selected = ('cudnn' if self.timings.get('cudnn', float('inf')) < .9*self.timings['torch'] else 'torch')
        try:
            return self._run(self.selected, q, k, v)
        except RuntimeError as error:
            if self.mode != 'auto' or self.selected == 'torch':
                raise
            # Fallback only this attention operation, before the caller mutates
            # its residual. Never rerun a partly updated in-place transformer.
            self.fallback_reasons.append(str(error))
            self.selected = 'torch'
            torch.cuda.empty_cache()
            return self._run('torch', q, k, v)
