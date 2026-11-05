"""Opt-in synchronized stage timings; never reset application CUDA peak counters."""
from contextlib import contextmanager
import time
import torch


@contextmanager
def measure_stage(pipeline, name, device):
    if not pipeline.profile:
        yield
        return
    device = torch.device(device)
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    event = dict(stage=name, seconds=None, completed=False)
    pipeline.stage_timings.append(event)
    try:
        yield
        if cuda:
            torch.cuda.synchronize(device)
        event['completed'] = True
    finally:
        event['seconds'] = time.perf_counter()-started
        if cuda:
            event.update(allocated_bytes=torch.cuda.memory_allocated(device),
                         reserved_bytes=torch.cuda.memory_reserved(device),
                         cumulative_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
