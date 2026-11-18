#!/usr/bin/env python3
"""Full-precision, physical-time conversion audit with fixed-world mesh panels.

Inputs: saved source-parity predictions. No neural inference, timeline warping,
per-frame pelvis alignment or post-hoc ground correction is performed.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motius.motion.representation.rotation import rotation_6d_to_matrix
from motius.motion.retarget import smpl_soma
from motius.motion.retarget._hml263_smpl_impl import recover_from_ric
from motius.motion.retarget.hml263_smpl import retarget_hml263_clip
from tools.smpl_audit_rig import GallerySMPL
from tools.validate_hml263_motion272 import forward_kinematics


def physical_sample(joints, source_fps, output_fps, count):
    # Independent np.interp oracle; deliberately not the production resampler.
    times = np.arange(count) / output_fps
    source_times = np.arange(len(joints)) / source_fps
    values = joints.reshape(len(joints), -1)
    return np.stack([np.interp(times, source_times, v) for v in values.T], -1).reshape(
        count, 22, 3
    )


def render(path, model, out_dir, fps):
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = np.load(path)
    vertices, source = data["vertices"], data["source_joints"]
    extent = vertices.reshape(-1, 3)
    lo, hi = extent.min(0), extent.max(0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2 + 0.15, 1.0)
    fig = plt.figure(figsize=(10.24, 5.12), dpi=100)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    surfaces = []
    lines = []
    for ax, azimuth in zip(axes, [45, 135]):
        ax.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[2] - radius, center[2] + radius),
            zlim=(min(0, float(lo[1])) - 0.1, max(2.0, float(hi[1]) + 0.1)),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=12, azim=azimuth)
        ax.set(xlabel="X (m)", ylabel="Z (m)", zlabel="Y (m)")
        xx, zz = np.meshgrid(
            [center[0] - radius, center[0] + radius],
            [center[2] - radius, center[2] + radius],
        )
        ax.plot_surface(xx, zz, np.zeros_like(xx), color="gray", alpha=0.15)
        surf = Poly3DCollection([], edgecolors="none")
        ax.add_collection3d(surf)
        surfaces.append(surf)
        lines.append(
            [ax.plot([], [], [], color="#ec791d", lw=1.5)[0] for _ in range(21)]
        )
    frames = []
    for t in range(0, len(vertices), 2):
        triangles = vertices[t][model.faces]
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        brightness = 0.4 + 0.6 * np.maximum(normal @ np.array([0.4, 0.8, 0.4]), 0)
        for surf, ls in zip(surfaces, lines):
            surf.set_verts(triangles[..., [0, 2, 1]])
            surf.set_facecolor(brightness[:, None] * np.array([0.25, 0.55, 0.8]))
            for j, line in enumerate(ls, 1):
                xyz = source[t, [int(model.parents[j]), j]]
                line.set_data(xyz[:, 0], xyz[:, 2])
                line.set_3d_properties(xyz[:, 1])
        fig.suptitle(
            f"{path.stem} | t={t / fps:.3f}s\nOrange: source; blue: SMPL. Fixed world and Y=0 ground; no alignment."
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    imageio.mimsave(
        out_dir / (path.stem + ".mp4"), frames, fps=fps / 2, codec="libx264"
    )
    imageio.imwrite(out_dir / (path.stem + ".png"), frames[len(frames) // 2])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "outputs/validation/native_method_architecture",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model_dir = ROOT / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model"
    model = GallerySMPL(model_dir)
    rest = model.rest[:22].numpy()
    parents = model.parents[:22].numpy()
    rig = (model, rest, parents)
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    records = []
    sources = {}

    def load(relative):
        path = args.inputs / relative
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return torch.load(path, weights_only=True, map_location="cpu")

    def check(name, result, reference, fps):
        motion = result["motion_135"]
        r = rotation_6d_to_matrix(motion[:, 3:].reshape(-1, 22, 6), convention="row")
        fk, _ = forward_kinematics(r, motion[:, :3] + rest[0], offsets, parents)
        fitted = result["fitted_joints"]
        target = result["target_joints"]
        aa = (
            Rotation.from_matrix(r.reshape(-1, 3, 3))
            .as_rotvec()
            .reshape(-1, 22, 3)
            .astype(np.float32)
        )
        # Chunk skinning to cap temporary allocations independently of clip length.
        vertices = []
        with torch.no_grad():
            for start in range(0, len(motion), 16):
                batch = aa[start : start + 16]
                body = np.zeros((len(batch), 69), np.float32)
                body[:, :63] = batch[:, 1:].reshape(-1, 63)
                v = model(
                    betas=torch.zeros(len(batch), 10),
                    body_pose=torch.from_numpy(body),
                    global_orient=torch.from_numpy(batch[:, 0]),
                    transl=torch.from_numpy(motion[start : start + 16, :3]),
                )
                vertices.append(v.vertices.numpy())
        vertices = np.concatenate(vertices)
        steps = np.rad2deg(
            Rotation.from_matrix(
                (r[1:] @ r[:-1].transpose(0, 1, 3, 2)).reshape(-1, 3, 3)
            ).magnitude()
        )
        foot_ids = [7, 8, 10, 11]
        srcfeet = reference[:, foot_ids]
        dstfeet = fitted[:, foot_ids]
        srcspeed = np.linalg.norm(np.diff(srcfeet[..., [0, 2]], axis=0), axis=-1) * fps
        dstspeed = np.linalg.norm(np.diff(dstfeet[..., [0, 2]], axis=0), axis=-1) * fps
        low = srcfeet[..., 1].min(axis=0) + 0.04
        contact = (
            (srcfeet[:-1, :, 1] < low) & (srcfeet[1:, :, 1] < low) & (srcspeed < 0.1)
        )
        row = {
            "name": name,
            "frames": len(motion),
            "fps": fps,
            "duration_s": len(motion) / fps,
            "mpjpe_mm": float(
                np.linalg.norm(fitted - reference, axis=-1).mean() * 1000
            ),
            "target_vs_physical_source_max_mm": float(
                np.linalg.norm(target - reference, axis=-1).max() * 1000
            ),
            "root_error_max_mm": float(
                np.linalg.norm(fitted[:, 0] - reference[:, 0], axis=-1).max() * 1000
            ),
            "serialization_fk_max_mm": float(
                np.linalg.norm(fk - fitted, axis=-1).max() * 1000
            ),
            "rotation_step_p99_deg": float(np.percentile(steps, 99)),
            "rotation_step_max_deg": float(steps.max()),
            "mesh_min_y_m": float(vertices[..., 1].min()),
            "source_joint_min_y_m": float(reference[..., 1].min()),
            "contact_intervals": int(contact.sum()),
            "source_contact_speed_mps": float(srcspeed[contact].mean())
            if contact.any()
            else None,
            "output_contact_speed_mps": float(dstspeed[contact].mean())
            if contact.any()
            else None,
        }
        row["mesh_floor_delta_y"] = float(
            result.get("mesh_floor_delta_y", result.get("height_align_delta_y", 0))
        )
        if not np.isfinite(vertices).all() or row["serialization_fk_max_mm"] > 0.01:
            raise AssertionError(f"{name}: invalid mesh or inconsistent serialization")
        np.savez_compressed(
            out / (name + ".npz"),
            source_joints=reference,
            fitted_joints=fitted,
            motion_135=motion,
            vertices=vertices,
            source_times=np.arange(len(motion)) / fps,
        )
        records.append(row)
        print(json.dumps(row), flush=True)

    def hml(name, features):
        result = retarget_hml263_clip(
            features, smpl_rest=rig, device="cpu", source_fps=20, target_fps=30
        )
        reference = physical_sample(
            recover_from_ric(features), 20, 30, len(result["motion_135"])
        )
        check(name, result, reference, 30)
        assert records[-1]["target_vs_physical_source_max_mm"] < 0.01
        assert records[-1]["root_error_max_mm"] < 0.01
        grounded = retarget_hml263_clip(
            features,
            smpl_rest=rig,
            device="cpu",
            source_fps=20,
            target_fps=30,
            floor_align=True,
        )
        check(name + "_grounded", grounded, reference, 30)
        assert abs(records[-1]["mesh_min_y_m"]) < 1e-5
        np.testing.assert_allclose(
            np.diff(grounded["transl"], axis=0),
            np.diff(result["transl"], axis=0),
            atol=1e-6,
        )

    data = load(Path("momask/native.pt"))
    for key, lengths in [("batch1_gumbelFalse", [40]), ("batch2_gumbelTrue", [60, 40])]:
        for i, length in enumerate(lengths):
            hml(f"momask_{key}_{i}", data[key]["physical"][i, :length].numpy())
    data = load(Path("mdm_inference_source/native.pt"))
    for c, record in enumerate(data["inference"]):
        for i, features in enumerate(record["output"]):
            hml(f"mdm_{c}_{i}", features.numpy())
    os.environ["KIMODO_SKELETON_ASSETS"] = str(
        ROOT / "motius/configuration/kimodo/assets/skeletons"
    )
    data = load(Path("kimodo_text_migration/native.pt"))
    with patch.object(smpl_soma, "_load_smpl_rest", return_value=rig):
        retarget = smpl_soma.KIMODOSOMAToSMPLRetargeter(
            model_dir="gallery_export", device="cpu"
        )
    for key in [
        "single",
        "batch",
        "multiprompt",
        "multiprompt_postprocess",
        "root_control",
    ]:
        for i in range(len(data[key]["posed_joints"])):
            soma = data[key]["posed_joints"][i].numpy()
            target = soma[:, smpl_soma.SOMA30_IN_SOMA77][:, smpl_soma.SMPL22_TO_SOMA30]
            for route in ["positions", "rotations"]:
                if route == "positions":
                    result = retarget.retarget_positions(target, soma)
                else:
                    result = retarget.retarget_rotations(
                        data[key]["global_rot_mats"][i].numpy(),
                        positions22=target,
                        root_positions=data[key]["root_positions"][i].numpy(),
                        soma77=soma,
                    )
                check(f"kimodo_{key}_{i}_{route}", result, target, 30)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "sources_sha256": sources,
                "rig_files_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in model_dir.iterdir()
                    if p.is_file()
                },
                "results": records,
            },
            indent=2,
        )
    )
    if args.render:
        for name in [
            "momask_batch2_gumbelTrue_1_grounded",
            "mdm_0_1_grounded",
            "kimodo_multiprompt_0_rotations",
            "kimodo_multiprompt_0_positions",
        ]:
            render(out / (name + ".npz"), model, out, 30)


if __name__ == "__main__":
    main()
