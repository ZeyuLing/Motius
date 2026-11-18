#!/usr/bin/env python3
"""Validate position IK on real motion against an optional saved 272 baseline.

Example:
  python tools/validate_hml263_motion272.py \
    --input tests/assets/humanml3d/004822_protocol.npz --output-dir outputs/hml272 --no-render

Rendering uses the repository's exported neutral SMPL web model, including
its rank-64 pose correctives and four skinning weights. It is the gallery
model, not a newly fitted body. Both sides use the same camera and pelvis
trajectory; MS272 pelvis positions are not treated as SMPL transl parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motius.motion.representation.convert import hml263_to_motion272
from motius.motion.representation.motion272 import (
    _canonical_272_offsets,
    recover_272_stored_positions,
    recover_local_rotations_and_root,
)
from motius.motion.skeleton.names import SMPL22_PARENTS


def forward_kinematics(rotations, root, offsets, parents):
    """Independent NumPy outgoing-frame FK for validation and mesh LBS."""
    world = np.zeros_like(rotations)
    positions = np.zeros(rotations.shape[:2] + (3,))
    world[:, 0], positions[:, 0] = rotations[:, 0], root
    for joint in range(1, len(parents)):
        parent = parents[joint]
        world[:, joint] = world[:, parent] @ rotations[:, joint]
        positions[:, joint] = positions[:, parent] + (world[:, parent] @ offsets[joint, :, None])[..., 0]
    return positions, world


def metrics(motion):
    positions = recover_272_stored_positions(motion)
    rotations, root = recover_local_rotations_and_root(motion)
    fk, _ = forward_kinematics(rotations, root, _canonical_272_offsets(), SMPL22_PARENTS)
    errors = np.linalg.norm(fk - positions, axis=-1) * 1000
    return {
        "frames": len(motion),
        "stored_vs_canonical_fk_mpjpe_mm": float(errors.mean()),
        "per_joint_mean_mm": errors.mean(axis=0).tolist(),
        "max_joint_frame_mm": float(errors.max()),
        "root_max_mm": float(errors[:, 0].max()),
    }


def web_mesh(motion, model_dir):
    metadata = json.loads((model_dir / "model.json").read_text())
    files = metadata["files"]
    vertices = np.fromfile(model_dir / files["vertices"], "<f4").reshape(-1, 3)
    rest = np.fromfile(model_dir / files["joints"], "<f4").reshape(-1, 3)
    faces = np.fromfile(model_dir / files["faces"], "<u2").reshape(-1, 3)
    indices = np.fromfile(model_dir / files["skin_indices"], "u1").reshape(-1, 4)
    weights = np.fromfile(model_dir / files["skin_weights"], "<u2").reshape(-1, 4).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    rotations22, root = recover_local_rotations_and_root(motion)
    rotations = np.tile(np.eye(3), (len(motion), 24, 1, 1))
    rotations[:, :22] = rotations22
    parents = metadata["parents"]
    offsets = rest.copy()
    offsets[1:] -= rest[np.asarray(parents[1:])]
    joints, world = forward_kinematics(rotations, root, offsets, parents)
    corrective = metadata.get("pose_correctives")
    if corrective:
        rank = corrective["rank"]
        basis = np.fromfile(model_dir / corrective["files"]["basis"], "<f2").reshape(rank, len(vertices), 4)[..., :3].astype(float)
        projection = np.fromfile(model_dir / corrective["files"]["projection"], "<f4").reshape(rank, -1)
        features = (rotations[:, 1:] - np.eye(3)).reshape(len(motion), -1)
        coefficients = features @ projection.T
    meshes = []
    for frame in range(len(motion)):
        posed = vertices.copy()
        if corrective:
            posed += np.einsum("r,rvc->vc", coefficients[frame], basis)
        transformed = np.einsum("vkij,vkj->vki", world[frame, indices], posed[:, None] - rest[indices])
        transformed += joints[frame, indices]
        meshes.append((transformed * weights[..., None]).sum(axis=1))
    return np.asarray(meshes), faces


def render_comparison(before, after, output_dir, model_dir, fps, stride):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial.transform import Rotation

    # Both bridges discard initial heading. Restore a common heading using
    # one rigid alignment of stored positions, shared across the entire clip.
    source_positions = recover_272_stored_positions(before)
    fixed_positions = recover_272_stored_positions(after)
    alignment = Rotation.align_vectors(source_positions.reshape(-1, 3), fixed_positions.reshape(-1, 3))[0]
    # Integrate full-rate velocities before sampling render frames.
    v_before, faces = web_mesh(before, model_dir)
    v_after, _ = web_mesh(after, model_dir)
    v_before = v_before[::stride]
    v_after = alignment.apply(v_after[::stride].reshape(-1, 3)).reshape(v_before.shape)
    all_vertices = np.concatenate((v_before, v_after))
    floor_y = float(all_vertices[..., 1].min())
    v_before[..., 1] -= floor_y
    v_after[..., 1] -= floor_y
    all_vertices[..., 1] -= floor_y
    low = all_vertices.min(axis=(0, 1))
    high = all_vertices.max(axis=(0, 1))
    center = (low + high) / 2
    extent = max(float((high - low).max()), 1.8) * 0.56
    fig = plt.figure(figsize=(9.6, 4.8), dpi=100, facecolor="#f6f7f8")
    colors = (np.array([0.78, 0.34, 0.23]), np.array([0.20, 0.51, 0.73]))
    axes, surfaces = [], []
    for panel, title in enumerate(("Before: incoming-frame mapping", "After: outgoing-frame position IK")):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        ax.set_facecolor("#f6f7f8")
        ax.set(xlim=(center[0]-extent, center[0]+extent),
               ylim=(center[2]-extent, center[2]+extent),
               zlim=(center[1]-extent, center[1]+extent))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=12, azim=57)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=0)
        for coordinate in np.linspace(-2, 2, 9):
            ax.plot([center[0]-2, center[0]+2], [center[2]+coordinate]*2, [0, 0], color="#d5d9dd", linewidth=0.45)
            ax.plot([center[0]+coordinate]*2, [center[2]-2, center[2]+2], [0, 0], color="#d5d9dd", linewidth=0.45)
        surface = Poly3DCollection([], linewidths=0, edgecolors="none", zsort="average")
        ax.add_collection3d(surface)
        axes.append(ax)
        surfaces.append(surface)
    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.91, wspace=0)
    caption = fig.text(0.5, 0.025, "", ha="center", fontsize=10, color="#40464b")
    light = np.array([0.4, 0.8, 0.5])
    light /= np.linalg.norm(light)
    frames = []
    try:
        for index, pair in enumerate(zip(v_before, v_after)):
            for vertices, surface, color in zip(pair, surfaces, colors):
                triangles = vertices[faces]
                normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                illumination = 0.38 + 0.62 * np.maximum(normals @ light, 0)
                surface.set_verts(triangles[..., [0, 2, 1]])
                surface.set_facecolor(illumination[:, None] * color)
            caption.set_text(f"Frame {index * stride} | Same neutral SMPL rig, camera and timeline")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if index == 0:
                imageio.imwrite(output_dir / "comparison.png", frames[-1])
            if (index + 1) % 25 == 0:
                print(f"Rendered {index + 1}/{len(v_before)} comparison frames", flush=True)
    finally:
        plt.close(fig)
    imageio.mimsave(output_dir / "comparison.mp4", frames, fps=fps / stride, codec="libx264", quality=8)
    imageio.mimsave(output_dir / "comparison.gif", frames, duration=1000 * stride / fps, loop=0)
    differences = np.linalg.norm(v_before - v_after, axis=-1).mean(axis=1)
    imageio.imwrite(output_dir / "comparison.png", frames[int(differences.argmax())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=20)
    parser.add_argument("--target-fps", type=float, default=30)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--baseline-motion272", type=Path, help="Saved .npy motion or .npz with before_272; no legacy algorithm is executed")
    parser.add_argument("--web-model-dir", type=Path, default=ROOT / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model")
    args = parser.parse_args()
    if not args.no_render and args.baseline_motion272 is None:
        parser.error("rendering a comparison requires --baseline-motion272; use --no-render for metrics only")
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            features = loaded["hml263"].copy()
    else:
        features = loaded
    args.output_dir.mkdir(parents=True, exist_ok=True)
    after = hml263_to_motion272(features, source_fps=args.source_fps, target_fps=args.target_fps)
    arrays = {"source_hml263": features, "after_272": after}
    before = None
    if args.baseline_motion272 is not None:
        baseline = np.load(args.baseline_motion272, allow_pickle=False)
        if isinstance(baseline, np.lib.npyio.NpzFile):
            with baseline:
                before = baseline["before_272"].copy()
        else:
            before = baseline
        if before.shape != after.shape or not np.isfinite(before).all():
            raise ValueError("baseline must be finite and have the same (T,272) shape as the current output")
        arrays["before_272"] = before
    np.savez_compressed(args.output_dir / "motions.npz", **arrays)
    report = {
        "source": str(args.input), "source_fps": args.source_fps, "target_fps": args.target_fps,
        "after": metrics(after),
        "baseline_path": str(args.baseline_motion272) if args.baseline_motion272 else None,
        "residual": "Native positions retain HumanML bone lengths/branch geometry; FK uses canon272. Not a lossless SMPL recovery.",
        "render_model": str(args.web_model_dir),
        "render_note": "Shared exported neutral SMPL rig with rank-64 pose correctives; common camera/floor, first-heading aligned; no mesh optimization.",
    }
    if before is not None:
        report["before"] = metrics(before)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.no_render:
        if args.stride < 1:
            parser.error("--stride must be positive")
        render_comparison(before, after, args.output_dir, args.web_model_dir, args.target_fps, args.stride)


if __name__ == "__main__":
    main()
