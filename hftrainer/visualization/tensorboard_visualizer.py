"""TensorBoard visualizer."""

import os
from typing import Dict, Any, Optional, List

from hftrainer.visualization.base_visualizer import BaseVisualizer
from hftrainer.registry import VISUALIZERS


@VISUALIZERS.register_module()
class TensorBoardVisualizer(BaseVisualizer):
    """
    Logs images and metrics to TensorBoard.

    Supports:
      - Image grids from output['preds'] (Tensor[B, C, H, W])
      - Text samples from output['preds'] + output['input_prompts'] (for LLM)
      - Scalar metrics
    """

    def __init__(
        self,
        log_dir: str = 'work_dirs/tb_logs',
        interval: int = 500,
        max_images: int = 8,
        task: str = 'generic',  # 'text2image', 'text2video', 'llm', 'classification'
        prompts: Optional[List[str]] = None,
    ):
        super().__init__(interval=interval)
        self.log_dir = log_dir
        self.max_images = max_images
        self.task = task
        self.prompts = prompts
        self._writer = None

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                from tensorboardX import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=self.log_dir)
        return self._writer

    def visualize(self, output: Dict[str, Any], step: int) -> None:
        writer = self._get_writer()

        # Log images
        if 'preds' in output:
            preds = output['preds']
            import torch
            if isinstance(preds, torch.Tensor) and preds.ndim == 4:
                # Tensor[B, C, H, W]
                imgs = preds[:self.max_images].clamp(0, 1)
                try:
                    from torchvision.utils import make_grid
                    grid = make_grid(imgs, nrow=min(4, len(imgs)))
                    writer.add_image('val/preds', grid, global_step=step)
                except ImportError:
                    # fallback: log first image
                    writer.add_image('val/pred_0', imgs[0], global_step=step)

        # Log text (LLM)
        if 'preds' in output and isinstance(output.get('preds'), list):
            preds = output['preds'][:4]
            gts = output.get('gts', [''] * len(preds))[:4]
            for i, (p, g) in enumerate(zip(preds, gts)):
                writer.add_text(f'val/pred_{i}', f"pred: {p}\ngt: {g}", global_step=step)

        writer.flush()
