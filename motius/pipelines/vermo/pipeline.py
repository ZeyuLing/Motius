"""VerMo motion-to-text inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from motius.motion.representation.rotation import repack_6d
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


def motion135_to_vermo138(motion135: np.ndarray) -> np.ndarray:
    """Convert ROW-major motion135 to VerMo's COLUMN-major abs-rel SMPL-22."""

    value = np.asarray(motion135, dtype=np.float32)
    if value.ndim != 2 or value.shape[-1] != 135:
        raise ValueError(f"Expected motion135 shape (T,135), got {value.shape}")
    translation = value[:, :3]
    velocity = np.zeros_like(translation)
    velocity[1:] = translation[1:] - translation[:-1]
    rotations = value[:, 3:].reshape(len(value), 22, 6)
    rotations = repack_6d(rotations, src="row", dst="column")
    return np.concatenate(
        [translation, velocity, rotations.reshape(len(value), 132)], axis=-1
    ).astype(np.float32)


@PIPELINES.register_module()
class VermoPipeline(BasePipeline):
    """Caption VerMo-138, motion135, or HumanML3D-263 motion inputs."""

    BUNDLE_CLS = "motius.models.vermo.VermoBundle"

    def __init__(
        self,
        bundle,
        *,
        smpl_model_dir: Optional[str] = None,
        hml263_rotation_init: str = "position_ik",
        **kwargs,
    ) -> None:
        super().__init__(bundle, **kwargs)
        self.smpl_model_dir = smpl_model_dir
        self.hml263_rotation_init = hml263_rotation_init
        self._smpl_rest = None

    @property
    def device(self) -> torch.device:
        return next(self.bundle.lm.parameters()).device

    def _hml263_to_vermo138(self, motion: np.ndarray) -> np.ndarray:
        from motius.motion.retarget.hml263_smpl import (
            hml263_to_motion135,
            load_smpl_rest,
        )

        if self._smpl_rest is None:
            self._smpl_rest = load_smpl_rest(
                self.smpl_model_dir,
                device=self.device,
                gender="neutral",
            )
        motion135 = hml263_to_motion135(
            motion,
            smpl_rest=self._smpl_rest,
            device=self.device,
            source_fps=20.0,
            target_fps=20.0,
            refine_iters=0,
            rotation_init=self.hml263_rotation_init,
        )
        return motion135_to_vermo138(motion135)

    def _prepare_motion(
        self,
        motion: Union[np.ndarray, torch.Tensor],
        length: int,
        input_representation: str,
    ) -> torch.Tensor:
        if isinstance(motion, torch.Tensor):
            value = motion.detach().cpu().numpy().astype(np.float32, copy=True)
        else:
            value = np.array(motion, dtype=np.float32, copy=True)
        value = value[: min(int(length), len(value))]
        representation = input_representation.lower().replace("-", "").replace("_", "")
        if representation in {"humanml3d263", "humanml263", "hml263"}:
            value = self._hml263_to_vermo138(value)
        elif representation in {"motion135", "smpl135"}:
            value = motion135_to_vermo138(value)
        elif representation not in {"vermo138", "smpl138"}:
            raise ValueError(f"Unsupported VerMo M2T input representation: {input_representation}")
        return torch.from_numpy(value).to(self.device)

    @torch.no_grad()
    def infer_m2t(
        self,
        motions: Sequence[Union[np.ndarray, torch.Tensor]],
        lengths: Optional[Sequence[int]] = None,
        *,
        input_representation: str = "humanml3d_263",
        max_new_tokens: int = 64,
    ) -> List[str]:
        from motius.models.vermo.task_utils.modality import Caption
        from motius.models.vermo.task_utils.task_lib.text_motion_tasks.motion2caption import (
            Motion2Caption,
        )

        if lengths is None:
            lengths = [len(motion) for motion in motions]
        if len(motions) != len(lengths):
            raise ValueError("motions and lengths must have equal length")

        tokenizer = self.bundle.processor.text_tokenizer
        eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        outputs: List[str] = []
        for motion, length in zip(motions, lengths):
            prepared = self._prepare_motion(
                motion, int(length), input_representation=input_representation
            )
            model_inputs = self.bundle.processor.process_train(
                {
                    "task": [Motion2Caption()],
                    "num_person": [1],
                    "motion": [prepared],
                    "caption": [""],
                }
            )
            input_ids = model_inputs["input_ids"][0]
            attention = model_inputs.get("attention_mask")
            if attention is not None:
                input_ids = input_ids[attention[0].bool()]
            positions = (input_ids == self.bundle.processor.output_bos_id).nonzero(
                as_tuple=False
            )
            if positions.numel() == 0:
                raise RuntimeError("VerMo prompt did not contain output BOS.")
            prefix = input_ids[: int(positions[-1].item()) + 1].unsqueeze(0)
            generated = self.bundle.lm.generate(
                input_ids=prefix,
                attention_mask=torch.ones_like(prefix),
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )[0, prefix.shape[1] :]
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
            matches = Caption.locate_modality(decoded)
            if matches:
                caption = matches[0].strip()
            else:
                caption = decoded
                if Caption.bos in caption:
                    caption = caption.split(Caption.bos, 1)[1]
                for marker in (Caption.eos, "<|eot_id|>"):
                    caption = caption.split(marker, 1)[0]
                caption = caption.strip()
            outputs.append(caption)
        return outputs

    def __call__(self, motions, lengths=None, **kwargs):
        return self.infer_m2t(motions, lengths=lengths, **kwargs)


__all__ = ["VermoPipeline", "motion135_to_vermo138"]
