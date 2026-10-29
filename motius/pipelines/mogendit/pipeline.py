"""Native Occam-201 motion repair with the recovered MoreDiff implementation."""
from __future__ import annotations

import math
import torch

from motius.models.mogendit import MoGenDiTBundle
from motius.pipelines.mogendit.refiner import MoreDiffRefiner
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MoGenDiTPipeline(BasePipeline):
    """Repair Y-up motion in metres, using global or local rotations per artifact.

    Input/output ``motion`` is (T,201) or (B,T,201). Channels are the archive's
    rotation-6D[22], pelvis-relative position[22], translation[3], not HML263 or
    HyMotion198. Results use the upstream canonical heading/origin/ground frame.
    ``restore_world=True`` aligns the first pelvis and heading back to the input.
    """
    BUNDLE_CLS = MoGenDiTBundle

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        from motius.motion.representation.mogendit.motion_representation import OccamMotionRep
        from motius.diffusion.mogendit import create_diffusion
        self.motion_rep = OccamMotionRep(global_pose=bundle.config["global_pose"], fps=bundle.config["fps"])
        self.diffusion = create_diffusion()

    def to(self, device):
        self.bundle.to(device)
        return self

    @torch.no_grad()
    def infer_motion_repair(self, motion, *, condition=None, mode='ada_denoise',
                            step=10, eta=1.0, imputation_mode='skip_last',
                            use_windowed=True, window_size=224, prev_padding=1,
                            fast_sampling=True, flat_ground_trans=False,
                            restore_world=False):
        if mode not in {'denoise', 'ada_denoise', 'trans_regen'}:
            raise ValueError(f"Unknown repair mode: {mode}")
        if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step < 1000:
            raise ValueError('step must be an integer in [1, 999]')
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError('eta must be finite and in [0, 1]')
        if imputation_mode not in {'skip_last', 'all', 'none'}:
            raise ValueError('imputation_mode must be skip_last, all or none')
        if (not isinstance(window_size, int) or not isinstance(prev_padding, int)
                or not 0 <= prev_padding < window_size <= 5000):
            raise ValueError('Require 0 <= prev_padding < window_size <= 5000')
        if flat_ground_trans and not use_windowed:
            raise ValueError('flat_ground_trans requires use_windowed=True')
        parameter = next(self.bundle.net.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError('The recovered MoreDiff runtime requires float32 weights')
        x = torch.as_tensor(motion, device=parameter.device, dtype=torch.float32)
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != 201 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError('motion must have shape (T,201) or (B,T,201), with B,T > 0')
        if not torch.isfinite(x).all():
            raise ValueError('motion must contain only finite values')
        if not use_windowed and x.shape[1] > 5000:
            raise ValueError('Sequences longer than 5000 frames require windowed repair')
        if flat_ground_trans and x.shape[1] < 2:
            raise ValueError('flat_ground_trans requires at least two frames')
        cond = None
        if condition is not None:
            cond = torch.as_tensor(condition, device=x.device, dtype=x.dtype)
            if cond.shape != (x.shape[0], 1, 66) or not torch.isfinite(cond).all():
                raise ValueError('condition must be finite with shape (B,1,66)')
        refiner = MoreDiffRefiner(self.motion_rep, self.bundle.net, self.diffusion)
        outputs = []
        for i in range(x.shape[0]):
            result = refiner.refine(
                x[i:i+1].clone(), None if cond is None else cond[i:i+1],
                mode=mode, step=step, eta=eta, imputation_mode=imputation_mode,
                use_windowed=use_windowed, window_size=window_size,
                prev_padding=prev_padding, fast_sampling=fast_sampling,
                flat_ground_trans=flat_ground_trans,
            )[0]
            if restore_world:
                result = self.motion_rep.pre_stitch(
                    result, x[i, :1], stitch_joint_idx=0, reset_height=False)
            outputs.append(result)
        result = torch.stack(outputs)
        poses, joints, translations = zip(*(self.motion_rep.decode(m) for m in result))
        output = dict(motion=result, poses=torch.stack(poses), joints=torch.stack(joints),
                      transl=torch.stack(translations))
        if unbatched:
            output = {key: value[0] for key, value in output.items()}
        return output

    __call__ = infer_motion_repair
