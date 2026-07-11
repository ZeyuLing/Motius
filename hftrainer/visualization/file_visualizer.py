"""FileVisualizer: saves validation outputs to disk as files."""

import os
from typing import Dict, Any, Optional

import torch

from hftrainer.visualization.base_visualizer import BaseVisualizer
from hftrainer.registry import VISUALIZERS
from hftrainer.utils.logger import get_logger

logger = get_logger()


@VISUALIZERS.register_module()
class FileVisualizer(BaseVisualizer):
    """
    Saves validation outputs to disk under ``save_dir/step_{N}/``.

    Supports multiple tasks via auto-detection from output keys:
      - Classification: saves ``img_0_pred{P}_gt{G}.png``
      - Text-to-image: saves ``0_{prompt}.png``
      - LLM: saves ``samples.txt``
      - Text-to-video: saves ``0_frame_00.png`` (first frame per sample)

    Args:
        save_dir: Root directory for saved files. Defaults to ``work_dirs/vis``.
        max_samples: Maximum number of samples to save per step.
    """

    def __init__(
        self,
        save_dir: str = 'work_dirs/vis',
        interval: int = 1,
        max_samples: int = 8,
    ):
        super().__init__(interval=interval)
        self.save_dir = save_dir
        self.max_samples = max_samples

    def visualize(self, output: Dict[str, Any], step: int) -> None:
        out_dir = os.path.join(self.save_dir, f'step_{step}')
        os.makedirs(out_dir, exist_ok=True)

        # Auto-detect task from output keys
        if 'scores' in output and 'gts' in output:
            self._vis_classification(output, out_dir)
        elif 'input_prompts' in output and isinstance(output.get('preds'), list):
            self._vis_llm(output, out_dir)
        elif 'prompts' in output and 'preds' in output:
            preds = output['preds']
            if isinstance(preds, torch.Tensor):
                if preds.ndim == 5:
                    self._vis_text2video(output, out_dir)
                elif preds.ndim == 4:
                    self._vis_text2image(output, out_dir)
                else:
                    self._vis_generic(output, out_dir)
            else:
                self._vis_generic(output, out_dir)
        else:
            self._vis_generic(output, out_dir)

        logger.info(f"Saved validation outputs to: {out_dir}")

    def _vis_classification(self, output: dict, out_dir: str):
        """Save classification results: images with pred/gt labels in filename."""
        preds = output.get('preds')
        gts = output.get('gts')
        images = output.get('images')  # optional: Tensor[B,C,H,W]

        n = min(self.max_samples, len(preds) if preds is not None else 0)

        # Write a summary text file
        lines = []
        for i in range(n):
            p = preds[i].item() if hasattr(preds[i], 'item') else preds[i]
            g = gts[i].item() if hasattr(gts[i], 'item') else gts[i]
            correct = 'OK' if p == g else 'WRONG'
            lines.append(f"sample {i}: pred={p}  gt={g}  [{correct}]")

        with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
            f.write('\n'.join(lines))

        # Save images if provided
        if images is not None and isinstance(images, torch.Tensor) and images.ndim == 4:
            self._save_images(images[:n], out_dir, preds, gts)

    def _vis_text2image(self, output: dict, out_dir: str):
        """Save generated images with prompt in filename."""
        preds = output['preds']  # Tensor[B,C,H,W]
        prompts = output.get('prompts', [])
        n = min(self.max_samples, preds.shape[0])

        for i in range(n):
            prompt_slug = self._slugify(prompts[i]) if i < len(prompts) else 'unknown'
            img = preds[i].clamp(0, 1)
            path = os.path.join(out_dir, f'{i}_{prompt_slug}.png')
            self._save_tensor_as_image(img, path)

    def _vis_text2video(self, output: dict, out_dir: str):
        """Save first frame of each generated video."""
        preds = output['preds']  # Tensor[B,T,C,H,W]
        prompts = output.get('prompts', [])
        n = min(self.max_samples, preds.shape[0])

        for i in range(n):
            prompt_slug = self._slugify(prompts[i]) if i < len(prompts) else 'unknown'
            frame = preds[i, 0].clamp(0, 1)  # first frame
            path = os.path.join(out_dir, f'{i}_{prompt_slug}_frame0.png')
            self._save_tensor_as_image(frame, path)

    def _vis_llm(self, output: dict, out_dir: str):
        """Save LLM text predictions to a text file."""
        preds = output.get('preds', [])
        gts = output.get('gts', [])
        prompts = output.get('input_prompts', [])
        n = min(self.max_samples, len(preds))

        lines = []
        for i in range(n):
            lines.append(f"--- Sample {i} ---")
            if i < len(prompts):
                lines.append(f"Prompt: {prompts[i]}")
            lines.append(f"Pred: {preds[i]}")
            if i < len(gts):
                lines.append(f"GT:   {gts[i]}")
            lines.append('')

        with open(os.path.join(out_dir, 'samples.txt'), 'w') as f:
            f.write('\n'.join(lines))

    def _vis_generic(self, output: dict, out_dir: str):
        """Fallback: dump keys and shapes to a text file."""
        lines = []
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                lines.append(f"{k}: Tensor shape={list(v.shape)} dtype={v.dtype}")
            elif isinstance(v, list):
                lines.append(f"{k}: list len={len(v)}")
            else:
                lines.append(f"{k}: {type(v).__name__}")

        with open(os.path.join(out_dir, 'output_summary.txt'), 'w') as f:
            f.write('\n'.join(lines))

    # ── Helpers ──

    @staticmethod
    def _slugify(text: str, max_len: int = 40) -> str:
        """Convert text to a filesystem-safe slug."""
        slug = text.lower().strip()
        slug = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in slug)
        slug = slug.replace(' ', '_')
        return slug[:max_len]

    @staticmethod
    def _save_tensor_as_image(tensor: torch.Tensor, path: str):
        """Save a [C,H,W] float tensor as a PNG image."""
        try:
            from torchvision.utils import save_image
            save_image(tensor, path)
        except ImportError:
            # Fallback: manual save via PIL
            try:
                from PIL import Image
                import numpy as np
                arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                if arr.shape[2] == 1:
                    arr = arr.squeeze(2)
                Image.fromarray(arr).save(path)
            except ImportError:
                pass  # No image saving available

    def _save_images(self, images: torch.Tensor, out_dir: str, preds=None, gts=None):
        """Save classification images with pred/gt in filename."""
        for i in range(images.shape[0]):
            p = preds[i].item() if preds is not None and hasattr(preds[i], 'item') else '?'
            g = gts[i].item() if gts is not None and hasattr(gts[i], 'item') else '?'
            img = images[i].clamp(0, 1)
            path = os.path.join(out_dir, f'img_{i}_pred{p}_gt{g}.png')
            self._save_tensor_as_image(img, path)
