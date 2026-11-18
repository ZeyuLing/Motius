"""Render three comparable full-precision tracks with fixed scale and camera."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_review(paths, labels, output, fps=30.0, azimuth=80.0):
    clips = [dict(np.load(p)) for p in paths]
    counts = [len(c["source_joints"]) for c in clips]
    if len(set(counts)) != 1:
        raise ValueError("Review clips must have identical physical timestamps")
    parents = np.array(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    )
    root = Path(__file__).resolve().parents[1]
    faces = np.fromfile(
        root / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model/faces.u16",
        "<u2",
    ).reshape(-1, 3)
    cloud = np.concatenate(
        [c.get("vertices", c["source_joints"]).reshape(-1, 3) for c in clips]
    )
    lo, hi = cloud.min(0), cloud.max(0)
    center = (lo + hi) / 2
    width = max(hi[0] - lo[0] + 0.15, 0.8)
    depth = max(hi[2] - lo[2] + 0.15, 0.8)
    low_y = min(0.0, lo[1]) - 0.05
    high_y = hi[1] + 0.05
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
    surfaces = []
    lines = []
    for i, (clip, label) in enumerate(zip(clips, labels)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set(
            xlim=(center[0] - width / 2, center[0] + width / 2),
            ylim=(center[2] - depth / 2, center[2] + depth / 2),
            zlim=(low_y, high_y),
        )
        ax.set_box_aspect((width, depth, high_y - low_y))
        ax.view_init(elev=8, azim=azimuth)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_title(label, fontsize=13)
        x, z = np.meshgrid(
            np.linspace(center[0] - width / 2, center[0] + width / 2, 7),
            np.linspace(center[2] - depth / 2, center[2] + depth / 2, 7),
        )
        ax.plot_wireframe(x, z, np.zeros_like(x), color="#cccccc", linewidth=0.5)
        if "vertices" in clip:
            surface = Poly3DCollection([], linewidths=0, edgecolors="none")
            ax.add_collection3d(surface)
            surfaces.append(surface)
            lines.append([])
        else:
            surfaces.append(None)
            lines.append(
                [
                    ax.plot(
                        [], [], [], color="#cc5b24", lw=4, marker="o", markersize=3
                    )[0]
                    for _ in range(21)
                ]
            )
    fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.92, wspace=0)
    clock = fig.text(0.5, 0.025, "", ha="center", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for t in range(counts[0]):
            for index, (clip, surface, ls) in enumerate(zip(clips, surfaces, lines)):
                if surface is not None:
                    triangles = clip["vertices"][t][faces]
                    normals = np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    )
                    normals /= np.maximum(
                        np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8
                    )
                    brightness = 0.45 + 0.55 * np.maximum(
                        normals @ np.array([0.3, 0.6, 0.7]), 0
                    )
                    color = (
                        np.array([0.4, 0.55, 0.7])
                        if index == 1
                        else np.array([0.28, 0.65, 0.53])
                    )
                    surface.set_verts(triangles[..., [0, 2, 1]])
                    surface.set_facecolor(brightness[:, None] * color)
                else:
                    for j, line in enumerate(ls, 1):
                        points = clip["source_joints"][t, [parents[j], j]]
                        line.set_data(points[:, 0], points[:, 2])
                        line.set_3d_properties(points[:, 1])
            clock.set_text(
                f"t={t / fps:.3f}s | Same source, same physical time and fixed camera | Candidate under review"
            )
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
            if t == counts[0] // 2:
                imageio.imwrite(output.with_suffix(".png"), frame)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs=3, type=Path, required=True)
    parser.add_argument("--labels", nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=80.0)
    args = parser.parse_args()
    render_review(args.clips, args.labels, args.output, azimuth=args.azimuth)
