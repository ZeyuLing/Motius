"""Check legacy human metadata against independently captured original HY-Motion."""

from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["artifact", "qwen", "clip", "reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from motius.models.hymotion_t2m import HyMotionT2MBundle
    from motius.pipelines.hymotion_t2m import HyMotionT2MPipeline
    from verify_motionstreamer_source import hashes, digest
    from verify_prompthmr_video_parity import compare_full, tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    metadata = json.loads((args.artifact / "hymotion_t2m_config.json").read_text())
    text = metadata["config"]["text_encoder"].copy()
    assert text["torch_dtype"] == "bf16"
    assert "sentence_torch_dtype" not in text and "llm_torch_dtype" not in text
    text.update(
        llm_model_path=str(args.qwen),
        llm_tokenizer_path=str(args.qwen),
        sentence_emb_model_path=str(args.clip),
        sentence_emb_tokenizer_path=str(args.clip),
    )
    report = dict(
        complete=False,
        scope="Legacy released human Lite artifact with original erroneous uniform-BF16 metadata (no component precision fields). Ordinary single/batch raw-caption 50-step public inference now matches independently captured original Tencent HY-Motion exactly. All motion neural and available null-condition hashes checked. The two omitted special-source weights require complete official checkpoint conversion and are outside this legacy certificate.",
        tests=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    bundle = HyMotionT2MBundle.from_pretrained(
        str(args.artifact), device="cuda", text_encoder=text
    ).eval()
    source = torch.load(args.reference, weights_only=True)
    assert hashes(bundle.motion_transformer) == source["neural"]
    for name in ["null_vtxt_feat", "null_ctxt_input"]:
        assert digest(getattr(bundle, name)) == source["conditions"][name]
    captions = ["A person walks forward.", "A person turns and raises both arms."]
    try:
        for index, (batch, length, scale) in enumerate([(1, 24, 5.0), (2, 40, 1.0)]):
            pipeline = HyMotionT2MPipeline(bundle, text_guidance_scale=scale)
            output = pipeline.infer_text_to_motion(
                captions[:batch], length, seed_input=[71, 923][:batch]
            )
            compare_full(
                source["cases"][index],
                tensors({name: output[name] for name in source["cases"][index]}),
                "/case" + str(index),
                report["tests"],
            )
        assert (
            next(bundle._text_encoder.llm_text_encoder.parameters()).dtype
            == torch.bfloat16
        )
        assert (
            next(bundle._text_encoder.sentence_emb_text_encoder.parameters()).dtype
            == torch.float32
        )
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Legacy HY-Motion metadata corrected to original full inference", flush=True)


if __name__ == "__main__":
    main()
