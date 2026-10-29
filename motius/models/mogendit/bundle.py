"""MoreDiff checkpoint-compatible model bundle from the supplied MoGenDiT snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from motius.models.mogendit.network.more_diff import MoreDiff
from motius.diffusion.mogendit.base_diffusion import BetaSchedule
from motius.diffusion.mogendit.base_diffusion import ModelMeanType

SIZES = {"0.03B": (512, 8, 8), "0.1B": (768, 12, 12), "0.3B": (1024, 16, 18)}


@MODEL_BUNDLES.register_module()
class MoGenDiTBundle(ModelBundle):
    """MoreDiff network, cosine x0 diffusion and native Occam-201 representation.

    ``network_config`` supports explicit dimensions for custom checkpoints/tests.
    Construction without weights is intended for training, not useful inference.
    """

    def __init__(self, version="0.1B", global_pose=True, fps=30,
                 network_config=None, weights_path=None):
        super().__init__()
        if version not in SIZES:
            raise ValueError(f"Unknown MoreDiff version {version!r}; expected {tuple(SIZES)}")
        if not isinstance(global_pose, bool):
            raise ValueError("global_pose must be a boolean matching the training config")
        if fps <= 0:
            raise ValueError("fps must be positive")
        width, heads, layers = SIZES[version]
        config = dict(d_motion=201, d_model=width, d_cond=66, n_head=heads,
                      n_stack=layers, dropout=0.0, window_size=90)
        config.update(network_config or {})
        if config['d_motion'] != 201 or config['d_cond'] != 66:
            raise ValueError("MoGenDiT repair requires d_motion=201 and d_cond=66")
        if (config['n_head'] <= 0 or config['d_model'] <= 0 or config['n_stack'] <= 0
                or config['d_model'] % (2 * config['n_head'])):
            raise ValueError("MoreDiff requires positive layers/heads and an even head dimension")
        self.config = dict(version=version, global_pose=global_pose, fps=fps,
                           network_config=config)
        self.net = MoreDiff(**config)
        self._trainable_modules = ["net"]
        self._save_ckpt_modules = ["net"]
        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path):
        """Load original raw/``model`` state dictionaries, including EMA files."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        self.net.load_state_dict(state, strict=True)

    def forward(self, motion, timesteps, *, condition=None, keep_mask=None, padding_mask=None):
        if keep_mask is None:
            keep_mask = torch.zeros_like(motion)
        return self.net(dict(x_t=motion, cond=condition, mask=keep_mask,
                             padding_mask=padding_mask), timesteps)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, checkpoint=None,
                        use_ema=True, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Expected a local MoGenDiT artifact directory: {path}")
        config_path = path / 'mogendit_config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            default_weights = path / 'model.pt'
        else:
            args = json.loads((path / 'train_args.json').read_text())
            # Do not guess the rotation convention of a legacy checkpoint.
            config = dict(version=args['model_version'], global_pose=args['global_pose'],
                          fps=args.get('fps', 30))
            prefix = 'ema_model_' if use_ema else 'model_'
            candidates = [p for p in path.glob(prefix + '*.pth')
                          if p.stem[len(prefix):].isdigit()]
            default_weights = max(candidates, key=lambda p: int(p.stem[len(prefix):])) if candidates else None
        weights = Path(checkpoint) if checkpoint is not None else default_weights
        if checkpoint is not None and not weights.is_absolute():
            weights = path / weights
        if weights is None or not weights.is_file():
            raise FileNotFoundError(f"No requested checkpoint in {path}; use_ema={use_ema}")
        config.update(kwargs)
        return cls(**config, weights_path=weights)

    def save_pretrained(self, save_directory, **kwargs):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'mogendit_config.json').write_text(json.dumps(self.config, indent=2) + '\n')
        torch.save({'model': self.net.state_dict()}, path / 'model.pt')
        (path / 'model_index.json').write_text(json.dumps({
            'pipeline_class': 'motius.pipelines.mogendit.MoGenDiTPipeline',
            'bundle_class': 'motius.models.mogendit.MoGenDiTBundle',
            'tasks': ['motion_repair'],
        }, indent=2) + '\n')
