from pathlib import Path

import torch

from motius.datasets.text_motion import ManifestTextMotionDataset
from motius.registry import MODEL_BUNDLES, TRAINERS
from tools.audit_training_release import audit
from tools.train import _import_custom_modules


def test_training_release_privacy_audit():
    assert audit() == []


def test_training_configs_register_their_components():
    from mmengine.config import Config

    root = Path(__file__).resolve().parents[1]
    expected = {
        "configs/prism/train_prism.py": ("PRISMBundle", "PrismTrainer"),
        "configs/tmr/train_tmr_smpl22.py": ("TMRBundle", "TMRTrainer"),
        "configs/hymotion_t2m/train_hymotion_t2m.py": (
            "HyMotionT2MBundle",
            "HyMotionT2MTrainer",
        ),
    }
    for relative_path, (bundle_name, trainer_name) in expected.items():
        cfg = Config.fromfile(root / relative_path)
        _import_custom_modules(cfg)
        assert MODEL_BUNDLES.get(bundle_name) is not None
        assert TRAINERS.get(trainer_name) is not None


def test_manifest_dataset_crops_pads_and_loads_cached_features(tmp_path: Path):
    motion_dir = tmp_path / "motions"
    feature_dir = tmp_path / "text_features"
    motion_dir.mkdir()
    feature_dir.mkdir()
    motion = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3).numpy()
    torch.save(
        {
            "t5_text_embeds": torch.ones(2, 4),
            "t5_text_mask": torch.ones(2, dtype=torch.long),
        },
        feature_dir / "sample.pt",
    )
    import json
    import numpy as np

    np.save(motion_dir / "sample.npy", motion)
    (tmp_path / "train.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "motion": "motions/sample.npy",
                        "caption": "walk",
                        "text_features": "text_features/sample.pt",
                    }
                ]
            }
        )
    )
    dataset = ManifestTextMotionDataset(
        data_root=str(tmp_path),
        manifest="train.json",
        motion_dim=3,
        max_frames=8,
        max_text_length=4,
        training=False,
    )
    sample = dataset[0]
    assert sample["motion"].shape == (8, 3)
    assert sample["num_frames"].item() == 5
    assert sample["t5_text_embeds"].shape == (4, 4)
    assert sample["t5_text_mask"].tolist() == [1, 1, 0, 0]
