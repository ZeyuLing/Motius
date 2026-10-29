"""Offline current-weight export of the real decoder-only Qwen3/CLIP towers."""

from pathlib import Path
from types import SimpleNamespace
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from motius.models.hymotion_t2m.network.text_encoder import HYTextModel
    from motius.motion.conditioning.hymotion_text import prepare_hymotion_text
    from motius.configuration.hymotion_text import export_text_networks
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.worker == "export":
        config = dict(
            type="HYTextModel",
            llm_type="qwen3",
            max_length_llm=64,
            llm_model_path=str(args.qwen),
            llm_tokenizer_path=str(args.qwen),
            sentence_emb_model_path=str(args.clip),
            sentence_emb_tokenizer_path=str(args.clip),
        )
    else:
        config = json.loads(
            (args.output_dir / "exported/text_encoder_config.json").read_text()
        )
        for key in [
            "llm_model_path",
            "llm_tokenizer_path",
            "sentence_emb_model_path",
            "sentence_emb_tokenizer_path",
        ]:
            config[key] = str(args.output_dir / "exported" / config[key])
    constructor = {
        key: value
        for key, value in config.items()
        if key not in {"type", "prompt_template", "sentence_padding"}
    }
    network = HYTextModel(**constructor).cuda().eval()
    if args.worker == "export":
        with torch.no_grad():
            network.llm_text_encoder.embed_tokens.weight[7, 0].add_(0.125)
            network.sentence_emb_text_encoder.text_model.final_layer_norm.bias[0].add_(
                0.0625
            )
    assert not hasattr(network.llm_text_encoder, "lm_head")
    assert not hasattr(network.sentence_emb_text_encoder, "vision_model")
    values = network.encode(
        prepare_hymotion_text(
            config, ["A person walks forward.", "A person turns and raises both arms."]
        )
    )
    result = tensors(
        dict(
            neural=hashes(network),
            features=values,
            llm_dtype=str(next(network.llm_text_encoder.parameters()).dtype),
            sentence_dtype=str(
                next(network.sentence_emb_text_encoder.parameters()).dtype
            ),
        )
    )
    torch.save(result, args.output_dir / (args.worker + ".pt"))
    if args.worker == "export":
        destination = args.output_dir / "exported"
        saved = export_text_networks(
            SimpleNamespace(_text_encoder=network, _text_dtype=None),
            destination,
            config,
        )
        (destination / "text_encoder_config.json").write_text(
            json.dumps(saved, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["qwen", "clip", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["export", "cold"])
    args = parser.parse_args()
    for name in ["qwen", "clip", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        scope="Actual Qwen3 decoder/CLIP text towers with modified current embedding/LN weights, all used neural hashes and raw-caption features through offline cold component export. Mixed BF16 decoder/FP32 sentence dtypes persist via per-component saved-config auto dtype. No unused language/vision output heads. Shared HY-Motion/MotionCanvas export helper integration; not a motion-generation/source-training certificate.",
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["export", "cold"]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--qwen",
            str(args.qwen),
            "--clip",
            str(args.clip),
            "--output-dir",
            str(args.output_dir),
            "--worker",
            kind,
        ]
        subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            check=True,
        )
    try:
        compare_full(
            torch.load(args.output_dir / "export.pt", weights_only=True),
            torch.load(args.output_dir / "cold.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Current Qwen decoder/CLIP text weights, raw features and mixed dtypes exact",
        flush=True,
    )


if __name__ == "__main__":
    main()
