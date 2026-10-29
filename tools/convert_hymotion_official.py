"""Convert an official Tencent HY-Motion checkpoint without dropping conditions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "qwen", "clip", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--include-text-encoder", action="store_true")
    args = parser.parse_args()
    import torch
    import yaml
    from motius.models.hymotion_t2m import HyMotionT2MBundle

    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text())
    options = config["train_pipeline_args"]
    if not options.get("enable_ctxt_null_feat", True):
        raise ValueError("This converter requires the official context-null variant")
    bundle = HyMotionT2MBundle(
        motion_transformer={
            "type": "HunyuanMotionMMDiT",
            **config["network_module_args"],
        },
        text_encoder={
            "type": "HYTextModel",
            **options["text_encoder_cfg"],
            "sentence_emb_type": "clipl",
            "llm_torch_dtype": "bf16",
            "sentence_torch_dtype": "float32",
            "llm_model_path": str(args.qwen.resolve()),
            "llm_tokenizer_path": str(args.qwen.resolve()),
            "sentence_emb_model_path": str(args.clip.resolve()),
            "sentence_emb_tokenizer_path": str(args.clip.resolve()),
        },
        enable_special_game_feat=options["enable_special_game_feat"],
        losses_cfg=dict(
            velocity_weight=1.0,
            x1_weight=0.0,
            keypoints3d_weight=0.0,
            translation_weight=0.0,
        ),
        noise_scheduler_cfg=options["noise_scheduler_cfg"],
        infer_noise_scheduler_cfg=options["infer_noise_scheduler_cfg"],
    )
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )["model_state_dict"]
    # Allocate the complete statistics before the strict load. Every source
    # parameter, including both source conditions, must be consumed.
    bundle.mean = torch.empty_like(state["mean"])
    bundle.std = torch.empty_like(state["std"])
    bundle.load_state_dict(state, strict=True)
    bundle.save_pretrained(
        str(args.output), include_text_encoder=args.include_text_encoder
    )
    (args.output / "conversion_provenance.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.file_digest(
                    args.checkpoint.open("rb"), "sha256"
                ).hexdigest(),
                "source_tensor_count": len(state),
                "strict_load": True,
                "special_conditions_preserved": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
