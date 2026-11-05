#!/usr/bin/env python3
"""Compare native PyTorch against a separate, pinned official checkout.

The official source is loaded only in this verification tool, never at runtime.
Checks raw generation (including all valid boundary frames), not the interactive
Warp/TensorRT engine's endpoint correction or scene controller.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from motius import Pipeline
from motius.configuration.motionbricks_release import UPSTREAM_REVISION
from motius.models.motionbricks import MotionBricksReleaseBundle
from motius.pipelines.motionbricks import MotionBricksReleasePipeline
from motius.motion.conditioning.motionbricks_release import release_input_shapes


def reference_package(source):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(
            f"Expected pinned upstream commit {UPSTREAM_REVISION}, got {revision}"
        )
    folder = source / "tools/onnx"
    spec = importlib.util.spec_from_file_location(
        "motionbricks_reference",
        folder / "__init__.py",
        submodule_search_locations=[str(folder)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return folder


def make_inputs(rig, case, csv=None):
    batch = 1 if case == 0 else 2
    values = {
        k: np.ones(
            (batch, *shape),
            dtype=np.int32 if k.startswith(("has_", "allowed_")) else np.float32,
        )
        for k, shape in release_input_shapes(rig).items()
    }
    if rig == "g1":
        q = np.zeros((300, 36), dtype=np.float32)
        q[:, 2] = 0.8
        q[:, 6] = 1
        if csv:
            from scipy.spatial.transform import Rotation

            rows = np.loadtxt(csv, delimiter=",", skiprows=1)
            q = np.concatenate(
                [
                    rows[:, 1:4] * 0.01,
                    Rotation.from_euler("xyz", rows[:, 4:7], degrees=True).as_quat(),
                    np.deg2rad(rows[:, 7:]),
                ],
                -1,
            ).astype(np.float32)
        values["context_qpos"] = np.stack(
            [q[108 + i * 12 : 124 + i * 12 : 4] for i in range(batch)]
        )
        values["target_qpos"] = np.stack(
            [q[192 + i * 12 : 208 + i * 12 : 4] for i in range(batch)]
        )
    else:
        # Nondegenerate, deterministic synthetic poses in the published feature layout.
        values["global_root_values"][:] = [0, 1, 0, 1, 0]
        values["local_root_values"][:] = [0, 0, 0, 1]
        rng = np.random.default_rng(42 + case)
        values["local_poses"][..., :87] = rng.normal(0, 0.2, (batch, 8, 87))
        values["local_poses"][..., 87:] = np.tile([1, 0, 0, 0, 1, 0], 30)
        values["global_root_values"][:, 4:, 0] = 0.6
    if case in (1, 2):
        values["allowed_pred_num_tokens"][:] = 0
        values["allowed_pred_num_tokens"][0, 0] = 1
        values["allowed_pred_num_tokens"][1, -1] = 1
    if case == 2:
        for k in ("has_local_poses", "has_local_root_values", "has_global_root_values"):
            values[k][:, 5:] = 0
    if case == 3:
        # Also exercise heading/translation variation and intermediate allowed lengths.
        values["allowed_pred_num_tokens"][:] = 0
        values["allowed_pred_num_tokens"][:, 3:8] = 1
        if rig == "g1":
            values["target_qpos"][..., 0] += 0.25
        else:
            values["global_root_values"][:, 4:, 3:] = [np.cos(0.3), np.sin(0.3)]
    return {k: torch.from_numpy(v) for k, v in values.items()}


def run(source, checkpoints, export_root, csv=None, device="cpu"):
    reference_package(source)
    from motionbricks_reference.checkpoints import load_sources, checkpoint_root
    from motionbricks_reference.checksums import CHECKPOINTS
    from motionbricks_reference.models.g1.model import (
        build_g1,
        OUTPUT_NAMES as G1_OUTPUTS,
    )
    from motionbricks_reference.models.soma.model import (
        build_soma,
        OUTPUT_NAMES as SOMA_OUTPUTS,
    )

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for rig in ("g1", "soma"):
        release = CHECKPOINTS[rig]
        sources = load_sources(checkpoint_root(checkpoints / rig, release), release)
        reference = (build_g1 if rig == "g1" else build_soma)(sources)
        reference.to(device)
        for module in reference.modules():
            if hasattr(module, "_device"):
                module._device = torch.device(device)
        for name in ("capture_offsets", "standard_offsets"):
            if hasattr(reference, name):
                setattr(reference, name, getattr(reference, name).to(device))
        del sources
        native = MotionBricksReleasePipeline(
            MotionBricksReleaseBundle(checkpoints / rig, model=rig, device=device)
        )
        output_names = G1_OUTPUTS if rig == "g1" else SOMA_OUTPUTS
        for case in range(4):
            inputs = {k: v.to(device) for k, v in make_inputs(rig, case, csv).items()}
            with torch.inference_mode():
                expected = dict(zip(output_names, reference(**inputs)))
                actual = native.infer_raw(inputs)
            counts = expected["pred_num_tokens"]
            torch.testing.assert_close(
                actual["pred_num_tokens"], counts, rtol=0, atol=0, check_dtype=False
            )
            entry = dict(rig=rig, case=case, token_counts=counts.tolist(), tensors=[])
            for key in output_names:
                # Tail padding is not part of the public motion contract.
                pairs = (
                    [(actual[key], expected[key])]
                    if key == "pred_num_tokens"
                    else [
                        (actual[key][i, : 4 * int(c)], expected[key][i, : 4 * int(c)])
                        for i, c in enumerate(counts)
                    ]
                )
                for i, (a, b) in enumerate(pairs):
                    assert torch.isfinite(a).all() and torch.isfinite(b).all()
                    torch.testing.assert_close(
                        a, b, rtol=1e-5, atol=1e-5, check_dtype=False
                    )
                    entry["tensors"].append(
                        dict(
                            name=key,
                            batch=i,
                            shape=list(a.shape),
                            bitwise_equal=torch.equal(a, b),
                            max_abs_error=float((a.double() - b.double()).abs().max()),
                        )
                    )
            print(
                rig, case, max(v["max_abs_error"] for v in entry["tensors"]), flush=True
            )
            results.append(entry)
        destination = export_root / rig
        native.save_pretrained(destination)
        reloaded = Pipeline.from_pretrained(
            destination, bundle_kwargs={"device": device}
        )
        with torch.inference_mode():
            before = native.infer_raw(inputs)
            after = reloaded.infer_raw(inputs)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        del native, reference, reloaded
    return dict(
        upstream_commit=UPSTREAM_REVISION,
        torch_version=torch.__version__,
        device=device,
        gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
        tf32=False,
        complete=True,
        scope="Independent official PyTorch raw generation vs native five-root bundle/pipeline; strict official weight loading and native save/reload. Excludes endpoint correction, scene control, TensorRT and motion-quality evaluation.",
        g1_input="official CSV" if csv else "synthetic",
        soma_input="synthetic varied poses",
        tests=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints/models/motionbricks_safetensors"),
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("checkpoints/models/motionbricks_native"),
    )
    parser.add_argument("--g1-csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/architecture/validation/motionbricks-release-source.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(
        args.source, args.checkpoints, args.export_root, args.g1_csv, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
