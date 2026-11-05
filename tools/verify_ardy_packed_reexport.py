"""Verify re-exporting an ARDY artifact that already contains its actual text NN."""

from pathlib import Path
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from motius import Pipeline
    from verify_ardy_offline_text import run, hash_state
    from verify_prompthmr_video_parity import compare_full

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = Pipeline.from_pretrained(args.artifact)
    expected = run(pipeline)
    weights = hash_state(pipeline.bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "artifact"
    pipeline.save_pretrained(
        destination, include_text_encoder=True, copy_mode="hardlink"
    )
    notices = Path(pipeline.bundle.packed_text_path) / "licenses"
    for file in notices.rglob("*"):
        if file.is_file():
            assert (
                file.read_bytes()
                == (
                    destination / "text_encoders/licenses" / file.relative_to(notices)
                ).read_bytes()
            )
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    restored = Pipeline.from_pretrained(destination)
    actual = run(restored)
    assert weights == hash_state(restored.bundle)
    rows = []
    compare_full(expected, actual, "", rows)
    assert not any(name == "ardy" or name.startswith("ardy.") for name in sys.modules)
    report = dict(
        complete=True,
        scope="Re-export of the complete previously verified real 8B ARDY text artifact using only packed tokenizer, current neural tensors and notices. Fresh native Pipeline reconstruction; no original text-model directory read.",
        tests=rows,
        all_current_neural_tensors_sha256_equal=True,
        notices_preserved=True,
    )
    (args.output_dir / "source_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Packed current-weight re-export exact:", len(rows))


if __name__ == "__main__":
    main()
