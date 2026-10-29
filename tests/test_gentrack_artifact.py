"""GenTrack's automatic loader must preserve its own sampler and current NN."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def test_current_gentrack_roundtrip_in_fresh_process(tmp_path):
    from motius.models.gentrack import GenTrackG1Bundle
    from motius.pipelines.gentrack import GenTrackPipeline

    torch.set_num_threads(1)
    torch.manual_seed(71)
    bundle = GenTrackG1Bundle(
        motion_transformer=dict(
            type="HunyuanMotionMMDiT",
            input_dim=38,
            feat_dim=32,
            ctxt_input_dim=12,
            vtxt_input_dim=8,
            num_layers=3,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        ),
        motion_type="g1_29dof",
        vtxt_input_dim=8,
        ctxt_input_dim=12,
        sample_steps=3,
        sample_guidance=1.5,
    )
    with torch.no_grad():
        list(bundle.motion_transformer.parameters())[-1].add_(0.03125)
        bundle.null_vtxt_feat.add_(0.125)
    pipeline = GenTrackPipeline(bundle)
    features = dict(
        text_vec_raw=torch.randn(2, 1, 8),
        text_ctxt_raw=[torch.randn(4, 12), torch.randn(3, 12)],
        text_ctxt_raw_length=torch.tensor([4, 3]),
    )
    noise = torch.randn(2, 360, 38)
    reference = pipeline.infer_text_to_motion(
        ["prepared one", "prepared two"],
        [9, 7],
        initial_noise=noise,
        **features,
    )
    directory = tmp_path / "exported"
    pipeline.save_pretrained(directory, include_text_encoder=False)
    torch.save(dict(features=features, noise=noise), tmp_path / "inputs.pt")
    root = Path(__file__).resolve().parents[1]
    code = """
import sys, torch
torch.set_num_threads(1)
from motius import Pipeline
pipe = Pipeline.from_pretrained(sys.argv[1], bundle_kwargs={"device": "cpu"})
assert type(pipe).__name__ == "GenTrackPipeline"
assert pipe.bundle.sample_steps == 3 and pipe.bundle.sample_guidance == 1.5
data = torch.load(sys.argv[2], weights_only=True)
result = pipe.infer_text_to_motion(["prepared one", "prepared two"], [9,7], initial_noise=data["noise"], **data["features"])
torch.save(dict(normalized_motion=result["normalized_motion"], qpos=torch.as_tensor(result["qpos"]), lengths=result["lengths"], neural=pipe.bundle.motion_transformer.state_dict()), sys.argv[3])
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(directory),
            str(tmp_path / "inputs.pt"),
            str(tmp_path / "cold.pt"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])},
        check=True,
    )
    current = torch.load(tmp_path / "cold.pt", weights_only=True)
    assert torch.equal(current["normalized_motion"], reference["normalized_motion"])
    assert np.array_equal(current["qpos"].numpy(), reference["qpos"])
    assert torch.equal(current["lengths"], reference["lengths"])
    for name, value in bundle.motion_transformer.state_dict().items():
        assert torch.equal(current["neural"][name], value), name
