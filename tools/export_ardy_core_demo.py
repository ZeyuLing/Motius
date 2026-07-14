#!/usr/bin/env python3
"""Generate ARDY Core mesh demos and Core->SMPL mesh comparisons.

This tool runs entirely through Motius. It never imports an upstream ARDY
checkout. In environments without the gated LLM2Vec/Llama text encoder, pass
``--synthetic-text-feat`` to verify the released Core checkpoint, decoder,
Core skinning, and SMPL retarget visualization paths. Such outputs are not
semantic text-to-motion results and are always written to a smoke-test output
directory unless ``--allow-synthetic-release-output`` is explicitly set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio.v2 as imageio
import numpy as np
import pyrender
import smplx
import torch
import trimesh

# Legacy SMPL/chumpy pickles still import these numpy aliases.
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

# Some locally mirrored SMPL pickles were serialized with numpy 2.x module
# names but are loaded in python environments pinned to numpy 1.x.
import numpy.core as _np_core  # noqa: E402
import numpy.core.multiarray as _np_multiarray  # noqa: E402
import numpy.core.umath as _np_umath  # noqa: E402

sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
sys.modules.setdefault("numpy._core.umath", _np_umath)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.ardy.network.skeleton import CoreSkeleton27  # noqa: E402
from motius.models.ardy.network.skeleton.kinematics import fk  # noqa: E402
from motius.motion import ardy_core27_to_smpl22_joints  # noqa: E402
from motius.motion.retarget.hml263_smpl import retarget_hml263_clip  # noqa: E402
from motius.pipelines.ardy import ARDYPipeline  # noqa: E402


DEFAULT_CASES = [
    {
        "sample_id": "rigplay_walk_turn_jog",
        "caption": "a person walks forward, turns right, and starts jogging",
    },
    {
        "sample_id": "rigplay_side_step",
        "caption": "a person quickly sidesteps to the left and then returns to standing",
    },
    {
        "sample_id": "rigplay_crouch_reach",
        "caption": "a person crouches down and reaches forward with both hands",
    },
]

SYNTHETIC_NOTE = (
    "Synthetic text features validate checkpoint/decoder/Core skinning/SMPL retarget plumbing only; "
    "they are random hash-conditioned tensors and are not semantic T2M results."
)


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _synthetic_text_inputs(caption: str, device: torch.device, tokens: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(_seed_from_text(caption))
    feat = torch.randn(1, tokens, 4096, generator=generator, device=device)
    mask = torch.ones(1, tokens, dtype=torch.bool, device=device)
    return feat, mask


def _load_cases(path: Path | None):
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("cases", payload.get("preview_cases", []))
    cases = []
    for item in payload:
        cases.append(
            {
                "sample_id": str(item.get("sample_id", item.get("id", len(cases)))),
                "caption": str(item["caption"]),
            }
        )
    if not cases:
        raise ValueError(f"no cases found in {path}")
    return cases


def _skin_asset_path() -> Path:
    return (
        ROOT
        / "motius"
        / "models"
        / "ardy"
        / "network"
        / "assets"
        / "skeletons"
        / "cskel27"
        / "skin_standard.npz"
    )


def core_lbs_vertices(
    local_rot_mats: np.ndarray,
    root_positions: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Skin ARDY Core-27 rotations with the vendored Core mesh asset."""

    skin_path = _skin_asset_path()
    if not skin_path.exists():
        raise FileNotFoundError(f"Core skin asset not found: {skin_path}")
    skin = np.load(skin_path, allow_pickle=True)
    bind_vertices = np.asarray(skin["bind_vertices"], dtype=np.float32)
    faces = np.asarray(skin["faces"], dtype=np.int32)
    bind = np.asarray(skin["bind_rig_transform"], dtype=np.float32)
    lbs_indices = np.asarray(skin["lbs_indices"], dtype=np.int64)
    lbs_weights = np.asarray(skin["lbs_weights"], dtype=np.float32)

    local = np.asarray(local_rot_mats, dtype=np.float32)
    root = np.asarray(root_positions, dtype=np.float32)
    if local.ndim == 5:
        local = local[0]
    if root.ndim == 3:
        root = root[0]
    if local.shape[-3:] != (27, 3, 3):
        raise ValueError(f"local_rot_mats must end with (27, 3, 3), got {local.shape}")
    if root.shape[-1] != 3 or root.shape[0] != local.shape[0]:
        raise ValueError(f"root_positions must have shape (T, 3), got {root.shape}")

    skeleton = CoreSkeleton27()
    device_t = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    with torch.no_grad():
        global_rots, posed_joints, _ = fk(
            torch.from_numpy(local).to(device_t),
            torch.from_numpy(root).to(device_t),
            skeleton,
        )
    global_rots_np = global_rots.detach().cpu().numpy().astype(np.float32)
    posed_joints_np = posed_joints.detach().cpu().numpy().astype(np.float32)

    transforms = np.tile(np.eye(4, dtype=np.float32), (local.shape[0], 27, 1, 1))
    transforms[:, :, :3, :3] = global_rots_np
    transforms[:, :, :3, 3] = posed_joints_np
    skin_transforms = transforms @ np.linalg.inv(bind)[None]

    vertex_h = np.concatenate([bind_vertices, np.ones((len(bind_vertices), 1), dtype=np.float32)], axis=1)
    gathered_t = skin_transforms[:, lbs_indices]
    transformed = np.einsum("tvkij,vj->tvki", gathered_t, vertex_h)
    vertices = (transformed[..., :3] * lbs_weights[None, :, :, None]).sum(axis=2).astype(np.float32)
    vertices[..., 1] -= float(vertices[..., 1].min())
    return vertices, faces


def _fit_smpl_vertices(
    smpl22_joints: np.ndarray,
    *,
    model_dir: Path,
    device: str,
    fps: int,
    refine_iters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    fit = retarget_hml263_clip(
        smpl22_joints,
        model_dir=model_dir,
        device=device,
        source_fps=fps,
        target_fps=fps,
        refine_iters=refine_iters,
        floor_align=False,
        rotation_init="position_ik",
    )
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = smplx.create(
        str(model_dir),
        model_type="smpl",
        gender="neutral",
        batch_size=1,
        use_pca=False,
    ).to(device_t)
    model.eval()
    faces = np.asarray(model.faces, dtype=np.int32)
    global_orient = np.asarray(fit["global_orient"], dtype=np.float32).reshape(-1, 3)
    body_pose = np.asarray(fit["body_pose"], dtype=np.float32).reshape(len(global_orient), -1)
    transl = np.asarray(fit["transl"], dtype=np.float32).reshape(len(global_orient), 3)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(global_orient), 96):
            end = min(start + 96, len(global_orient))
            count = end - start
            body69 = np.zeros((count, 69), dtype=np.float32)
            body69[:, : body_pose.shape[1]] = body_pose[start:end, :69]
            out = model(
                betas=torch.zeros(count, 10, device=device_t),
                body_pose=torch.from_numpy(body69).to(device_t),
                global_orient=torch.from_numpy(global_orient[start:end]).to(device_t),
                transl=torch.from_numpy(transl[start:end]).to(device_t),
            )
            chunks.append(out.vertices.detach().cpu().numpy().astype(np.float32))
    vertices = np.concatenate(chunks, axis=0)
    vertices[..., 1] -= float(vertices[..., 1].min())
    metrics = {"fit_mpjpe_mm": float(np.asarray(fit["fit_mpjpe_mm"]).mean())}
    return vertices, faces, metrics


def _center_and_offset(vertices: np.ndarray, x_offset: float) -> np.ndarray:
    out = np.asarray(vertices, dtype=np.float32).copy()
    origin = out[0].mean(axis=0)
    origin[1] = 0.0
    out -= origin
    out[..., 1] -= float(out[..., 1].min())
    out[..., 0] += x_offset
    return out


def _look_at(eye: np.ndarray, center: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = center - eye
    forward /= np.linalg.norm(forward) + 1e-9
    side = np.cross(forward, up)
    side /= np.linalg.norm(side) + 1e-9
    true_up = np.cross(side, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = side
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def render_core_smpl_mesh_gif(
    core_vertices: np.ndarray,
    core_faces: np.ndarray,
    smpl_vertices: np.ndarray,
    smpl_faces: np.ndarray,
    path: Path,
    *,
    fps: int,
    width: int,
    height: int,
    max_frames: int,
) -> list[np.ndarray]:
    """Render Core mesh and SMPL mesh side by side."""

    core = _center_and_offset(core_vertices[:max_frames], -0.8)
    smpl = _center_and_offset(smpl_vertices[: len(core)], 0.8)
    all_pts = np.concatenate([core.reshape(-1, 3), smpl.reshape(-1, 3)], axis=0)
    y_center = float(np.percentile(all_pts[:, 1], 54))
    span = np.ptp(all_pts, axis=0)
    radius = max(float(span.max()), 1.8)
    dist = max(3.0, radius * 1.45)
    target = np.array([0.0, y_center, float(np.mean(all_pts[:, 2]))], dtype=np.float32)
    eye = target + np.array([0.72 * dist, 0.42 * dist, 1.05 * dist], dtype=np.float32)
    camera_pose = _look_at(eye, target)

    materials = {
        "core": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.05, 0.63, 0.70, 1.0), metallicFactor=0.02, roughnessFactor=0.62
        ),
        "smpl": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.92, 0.36, 0.22, 1.0), metallicFactor=0.02, roughnessFactor=0.58
        ),
        "floor": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.90, 0.91, 0.93, 1.0), metallicFactor=0.0, roughnessFactor=1.0
        ),
    }
    renderer = pyrender.OffscreenRenderer(width, height)
    frames: list[np.ndarray] = []
    try:
        for frame in range(len(core)):
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.48, 0.48, 0.52])
            scene.add(
                pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(core[frame], core_faces, process=False),
                    material=materials["core"],
                    smooth=True,
                )
            )
            scene.add(
                pyrender.Mesh.from_trimesh(
                    trimesh.Trimesh(smpl[frame], smpl_faces, process=False),
                    material=materials["smpl"],
                    smooth=True,
                )
            )
            floor_size = max(3.8, radius * 1.45)
            floor = trimesh.creation.box(extents=(floor_size, 0.012, floor_size))
            floor.apply_translation([0.0, -0.006, target[2]])
            scene.add(pyrender.Mesh.from_trimesh(floor, material=materials["floor"], smooth=False))
            scene.add(pyrender.PerspectiveCamera(yfov=math.radians(36), aspectRatio=width / height), pose=camera_pose)
            for light_eye, intensity in [
                (target + np.array([2.6, 4.2, 3.0], dtype=np.float32), 4.0),
                (target + np.array([-3.0, 2.8, 1.4], dtype=np.float32), 1.9),
                (target + np.array([0.0, 3.3, -3.2], dtype=np.float32), 1.5),
            ]:
                scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                    pose=_look_at(light_eye, target),
                )
            color, _ = renderer.render(scene)
            frames.append(color)
    finally:
        renderer.delete()
    if not frames:
        raise RuntimeError("mesh render produced zero frames")
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ardy/core_humanml3d_demo"))
    parser.add_argument("--checkpoint", default="core8")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--synthetic-text-feat", action="store_true")
    parser.add_argument(
        "--allow-synthetic-release-output",
        action="store_true",
        help=(
            "Allow --synthetic-text-feat to write to the requested output dir. "
            "Use only for debugging; synthetic outputs are not release demos."
        ),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("checkpoints/ardy"))
    parser.add_argument("--model-dir", type=Path, default=None, help="SMPL model directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refine-iters", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-render-frames", type=int, default=96)
    parser.add_argument("--write-mp4", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir or os.environ.get("MOTIUS_SMPL_MODEL_DIR")
    if not model_dir:
        raise FileNotFoundError("SMPL mesh comparison requires --model-dir or MOTIUS_SMPL_MODEL_DIR")
    model_dir = Path(model_dir)

    if args.synthetic_text_feat and not args.allow_synthetic_release_output:
        requested = args.output_dir
        if requested.name != "smoke":
            args.output_dir = requested / "smoke"
        print(
            f"--synthetic-text-feat is a smoke-test mode; writing to {args.output_dir} instead of {requested}.",
            flush=True,
        )

    if args.synthetic_text_feat:
        args.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(args.hf_home.resolve()))

    bundle_kwargs = {"device": args.device}
    if args.synthetic_text_feat:
        bundle_kwargs["text_encoder"] = False
    pipe = ARDYPipeline.from_pretrained(args.checkpoint, bundle_kwargs=bundle_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": pipe.bundle.model_name,
        "frames": args.frames,
        "fps": args.fps,
        "synthetic_text_feat": bool(args.synthetic_text_feat),
        "release_ready": not bool(args.synthetic_text_feat),
        "smpl_model_dir": str(model_dir),
        "note": SYNTHETIC_NOTE if args.synthetic_text_feat else "Official text encoder path.",
        "cases": [],
    }
    for index, case in enumerate(_load_cases(args.cases)):
        caption = case["caption"]
        kwargs = {}
        if args.synthetic_text_feat:
            text_feat, text_mask = _synthetic_text_inputs(caption, pipe.device, args.tokens)
            kwargs.update(text_feat=text_feat, text_pad_mask=text_mask)
        result = pipe.text_to_motion(
            caption,
            args.frames,
            num_denoising_steps=args.steps,
            seed=args.seed + index,
            return_numpy=True,
            **kwargs,
        )
        core_joints = np.asarray(result["posed_joints"][0, : args.frames], dtype=np.float32)
        smpl22_joints = ardy_core27_to_smpl22_joints(core_joints)
        core_vertices, core_faces = core_lbs_vertices(
            np.asarray(result["local_rot_mats"][0, : args.frames], dtype=np.float32),
            np.asarray(result["root_positions"][0, : args.frames], dtype=np.float32),
            device=args.device,
        )
        smpl_vertices, smpl_faces, smpl_metrics = _fit_smpl_vertices(
            smpl22_joints,
            model_dir=model_dir,
            device=args.device,
            fps=args.fps,
            refine_iters=args.refine_iters,
        )
        stem = f"ardy_core_{case['sample_id']}"
        npz_path = args.output_dir / f"{stem}_core_smpl_mesh.npz"
        gif_path = args.output_dir / f"{stem}_core_mesh_smpl_mesh.gif"
        np.savez_compressed(
            npz_path,
            core_joints=core_joints,
            smpl22_joints=smpl22_joints,
            core_vertices=core_vertices,
            core_faces=core_faces,
            smpl_vertices=smpl_vertices,
            smpl_faces=smpl_faces,
            caption=caption,
            sample_id=case["sample_id"],
            fps=args.fps,
            fit_mpjpe_mm=smpl_metrics["fit_mpjpe_mm"],
        )
        frames = render_core_smpl_mesh_gif(
            core_vertices,
            core_faces,
            smpl_vertices,
            smpl_faces,
            gif_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_frames=args.max_render_frames,
        )
        case_meta = {
            "sample_id": case["sample_id"],
            "caption": caption,
            "npz": str(npz_path),
            "gif": str(gif_path),
            "fit_mpjpe_mm": smpl_metrics["fit_mpjpe_mm"],
        }
        if args.write_mp4:
            mp4_path = args.output_dir / f"{stem}_core_mesh_smpl_mesh.mp4"
            imageio.mimwrite(mp4_path, frames, fps=args.fps, quality=8, macro_block_size=1)
            case_meta["mp4"] = str(mp4_path)
        manifest["cases"].append(case_meta)
        print(json.dumps(case_meta), flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
