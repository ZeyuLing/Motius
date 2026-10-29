"""Check MotionCLIP NN/loss relocation against pre-migration code.

Synthetic tiny architecture and original snapshot, not released-weight or
independent upstream certification. Runs the actual snapshot neural/loss code.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = args.output_dir / "oracle" / "motionclip_snapshot"
    source.mkdir(parents=True, exist_ok=True)
    (source / "__init__.py").write_text("")
    hashes = {}
    for name in [
        "configuration_motionclip",
        "modeling_motionclip",
        "modeling_motionclip_base",
        "modeling_motionclip_text",
        "modeling_motionclip_motion",
    ]:
        path = args.snapshot / (name + ".py")
        shutil.copy2(path, source / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sys.path.insert(0, str(source.parent))
    configuration = importlib.import_module(
        "motionclip_snapshot.configuration_motionclip"
    )
    original_model = importlib.import_module(
        "motionclip_snapshot.modeling_motionclip"
    ).MotionCLIPModel
    from motius.models.motion_clip import MotionCLIPBundle
    from motius.trainers.motion_clip import MotionCLIPTrainer
    from types import SimpleNamespace

    report = dict(
        complete=False,
        scope=__doc__,
        snapshot="pre-migration source snapshot",
        device=args.device,
        source_files=hashes,
        comparisons=[],
    )

    def compare(left, right, name):
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
            report["comparisons"].append(
                dict(tensor=name, shape=list(left.shape), bitwise_equal=True)
            )

    text = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=12,
        bos_token_id=30,
        eos_token_id=31,
    )
    motion = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=12,
        motion_dim=6,
    )
    for training, freeze, batch in [
        (True, False, 3),
        (True, True, 3),
        (False, False, 3),
        (True, False, 1),
    ]:
        torch.manual_seed(293)
        config = configuration.MotionCLIPConfig(
            text_config=text, motion_config=motion, projection_dim=8
        )
        original = original_model(config).to(args.device).train(training)
        native = (
            MotionCLIPBundle(text, motion, projection_dim=8, freeze_text_encoder=freeze)
            .to(args.device)
            .train(training)
        )
        native.motionclip_model.load_state_dict(original.state_dict(), strict=True)
        if freeze:
            for module in [original.text_model, original.text_projection]:
                module.requires_grad_(False)
        ids = torch.randint(2, 29, (batch, 6), device=args.device)
        ids[:, -1] = 31
        values = torch.randn(batch, 7, 6, device=args.device)
        text_mask = torch.ones(batch, 6, device=args.device)
        motion_mask = torch.ones(batch, 7, device=args.device)
        if batch > 1:
            motion_mask[-1, -2:] = 0
            text_mask[-1, -1:] = 0
        inputs = dict(
            input_ids=ids,
            motion_values=values,
            attention_mask=text_mask,
            motion_attention_mask=motion_mask,
        )
        expected = original(**inputs, return_loss=True)
        actual = native(**inputs)
        loss = MotionCLIPTrainer.objective(SimpleNamespace(bundle=native), actual)
        prefix = f"train={training}/freeze={freeze}/batch={batch}/"
        for name in [
            "text_embeds",
            "motion_embeds",
            "logits_per_text",
            "logits_per_motion",
        ]:
            compare(getattr(expected, name), getattr(actual, name), prefix + name)
        compare(expected.loss, loss, prefix + "loss")
        expected.loss.backward()
        loss.backward()
        for name, value in original.named_parameters():
            compare(
                value.grad,
                dict(native.motionclip_model.named_parameters())[name].grad,
                prefix + "gradient/" + name,
            )
    report["complete"] = True
    (args.output_dir / "source_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Exact snapshot NN/loss comparisons:", len(report["comparisons"]))


if __name__ == "__main__":
    main()
