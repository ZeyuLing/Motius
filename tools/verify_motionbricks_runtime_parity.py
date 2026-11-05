#!/usr/bin/env python3
"""Compare complete native generation with the pinned official TensorRT engine.

Reference source and ONNX files remain outside the Motius package. Shared inputs
cover heading/world placement, 1/4 target poses, length extremes, correction on/off
and regeneration from played context. This test requires an NVIDIA GPU.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from motius import Pipeline


def run(source, assets, checkpoints, cache_dir):
    from tools.verify_motionbricks_release_parity import reference_package

    reference_package(source)
    sys.path.insert(0, str(source))
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.io.bvh import load_bvh as ref_bvh
    from motionbricks.io.csv import load_csv as ref_csv
    from motionbricks.model.config import MotionBricksInferenceConfig as RefConfig
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine
    from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine

    from motius.configuration.motionbricks_runtime import MotionBricksInferenceConfig
    from motius.configuration.motionbricks_runtime_contract import ModelRig
    from motius.motion.io.motionbricks.bvh import load_bvh
    from motius.motion.io.motionbricks.csv import load_csv
    from motius.pipelines.motionbricks.scene_assets import model_to_runtime

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wp.init()
    wp.set_device("cuda:0")
    results = []
    for rig in ["g1", "soma"]:
        folder = "unitree_g1" if rig == "g1" else "soma"
        clip_path = (
            assets
            / f"motions/{folder}/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.{'csv' if rig == 'g1' else 'bvh'}"
        )
        if rig == "g1":
            native_clip = load_csv(clip_path)
            reference_clip = ref_csv(clip_path)
        else:
            from motionbricks.io.usd import load_skeleton_from_usd as ref_skeleton

            from motius.motion.io.motionbricks.usd_skeleton import (
                load_skeleton_from_usd,
            )

            usd = assets / "human/soma_base_skel_minimal_no_root.usd"
            prim = "/OUTPUT/c_skeleton_grp/Hips"
            native_clip = load_bvh(clip_path, load_skeleton_from_usd(usd, prim))
            reference_clip = ref_bvh(clip_path, ref_skeleton(usd, prim))
        transform = model_to_runtime(
            ModelRig.UNITREE_G1 if rig == "g1" else ModelRig.SOMA
        )
        pipe = Pipeline.from_pretrained(
            checkpoints / rig, bundle_kwargs={"device": "cuda"}
        )
        native = pipe.create_runtime(model_to_runtime=transform)
        graph = source / f"motionbricks/assets/onnx/{folder}/mmm_v6_{rig}_fp32.onnx"
        reference = (
            MotionBricksUnitreeG1Engine if rig == "g1" else MotionBricksSOMAEngine
        )(
            graph,
            model_to_runtime=transform,
            build_options=TensorRTBuildOptions(
                enable_tf32=False,
                workspace_size_bytes=512 * 1024**2,
                builder_optimization_level=0,
            ),
            cache_dir=cache_dir,
            bake_cuda_graph=False,
        )
        print(rig, "both runtimes ready", flush=True)

        def buffers(engine, clip):
            args = () if rig == "g1" else (clip.skeleton,)
            return (
                engine.create_input(*args),
                engine.create_input(*args),
                engine.create_output(*args),
            )

        nc, nt, no = buffers(native, native_clip)
        rc, rt, ro = buffers(reference, reference_clip)
        shape = (4, *native_clip.poses.shape[1:])
        dtype = native_clip.poses.dtype
        context = wp.empty(shape, dtype=dtype, device=native.device)
        target = wp.empty_like(context)
        cases = [
            (6, 1, True),
            (16, 4, True),
            (10, 4, False),
            (8, 2, True),
            (12, 4, True),
        ]
        for case, (tokens, targets, correct) in enumerate(cases):
            # A nonzero world translation and yaw also exercise alignment reversal.
            offset = wp.mul(
                wp.transform(
                    wp.vec3(2.0, -1.0, 0.0),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4),
                ),
                transform,
            )
            for i in range(4):
                if case < 4:
                    native_clip.sample(
                        0.9 + i / 30,
                        context[i],
                        root_offset=offset,
                        stream=native.stream,
                    )
                else:
                    wp.copy(
                        context[i],
                        no.poses[no.num_frames - 4 + i],
                        stream=native.stream,
                    )
                native_clip.sample(
                    1.6 + i / 30, target[i], root_offset=offset, stream=native.stream
                )
            native.synchronize()
            for engine, c, t in [(native, nc, nt), (reference, rc, rt)]:
                c.update_from_poses(context, stream=engine.stream)
                t.update_from_poses(target[:targets], stream=engine.stream)
            kwargs = dict(
                allowed_token_counts=(tokens,), correct_generated_diff=correct
            )
            native.generate_animation(
                nc, nt, no, config=MotionBricksInferenceConfig(**kwargs)
            )
            reference.generate_animation(rc, rt, ro, config=RefConfig(**kwargs))
            a = no.poses.numpy()[: no.num_frames]
            b = ro.poses.numpy()[: ro.num_frames]
            assert no.num_frames == ro.num_frames == 4 * tokens - 6
            # Quaternions q and -q encode the same rotation.
            aa = a.copy()
            bb = b.copy()
            qa = aa[:, 3:7] if rig == "g1" else aa[..., 3:7]
            qb = bb[:, 3:7] if rig == "g1" else bb[..., 3:7]
            qb *= np.where((qa * qb).sum(-1, keepdims=True) < 0, -1, 1)
            error = float(np.max(np.abs(aa - bb)))
            contacts = float(
                np.max(
                    np.abs(
                        no.foot_contacts.numpy()[: no.num_frames]
                        - ro.foot_contacts.numpy()[: ro.num_frames]
                    )
                )
            )
            # Record errors before asserting: TensorRT may choose other discrete pose tokens.
            entry = dict(
                rig=rig,
                case=case,
                chained_context=case == 4,
                tokens=tokens,
                target_poses=targets,
                corrected=correct,
                pose_max_abs_error=error,
                contacts_max_abs_error=contacts,
            )
            print(entry, flush=True)
            results.append(entry)
            np.testing.assert_allclose(aa, bb, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(
                no.foot_contacts.numpy()[: no.num_frames],
                ro.foot_contacts.numpy()[: ro.num_frames],
                atol=5e-4,
                rtol=2e-4,
            )
        native.close()
        reference.close()
        del native, reference, pipe
        torch.cuda.empty_cache()
    return dict(
        complete=True,
        source_revision="54fc563aeaaf819a753d72cc9fdc6713989cdd43",
        pose_atol=2e-4,
        contacts_atol=5e-4,
        rtol=2e-4,
        scope="Native PyTorch + native pose runtime versus official TensorRT FP32 (TF32 disabled) + official pose runtime",
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        warp_version=wp.__version__,
        tests=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--assets",
        type=Path,
        default=Path("checkpoints/models/motionbricks_scene_assets"),
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints/models/motionbricks_native"),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/motionbricks_native_audit/trt_cache"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/architecture/validation/motionbricks-runtime-tensorrt.json"),
    )
    a = p.parse_args()
    r = run(a.source, a.assets, a.checkpoints, a.cache_dir)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
