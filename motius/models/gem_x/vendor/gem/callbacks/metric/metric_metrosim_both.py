# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metric callback for Metrosim dataset (SOMA body model).

Computes in-cam and global metrics using SOMA FK directly on predicted/GT params.
"""

import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from gem.utils.eval_utils import (
    as_np_array,
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_jitter,
    compute_jpe,
    compute_rte,
    first_align_joints,
    global_align_joints,
)
from gem.utils.gather import all_gather
from gem.utils.geo_transform import apply_T_on_points
from gem.utils.pylogger import Log
from gem.utils.soma_utils.soma_layer import SomaLayer


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build 4x4 camera-to-world pose matrix from look-at parameters (numpy)."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z = eye - target
    z_norm = np.linalg.norm(z)
    z = z / (z_norm + 1e-8)
    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x = np.cross(up, z)
        x_norm = np.linalg.norm(x)
    x = x / (x_norm + 1e-8)
    y = np.cross(z, x)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _compute_vertex_normals_np(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals (pure numpy)."""
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(positions)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


def _visualize_mesh_pyrender_mp4(
    pred_verts,
    gt_verts,
    soma_faces: np.ndarray,
    out_path: str,
    fps: int = 30,
    vis_max_frames: int = 150,
    image_size: int = 512,
    pyopengl_platform: str = "osmesa",
    egl_device_id=None,
):
    """Renders side-by-side pred|gt global mesh MP4 using pyrender.

    pred_verts: (L, V, 3) tensor or ndarray — predicted global vertices
    gt_verts:   (L, V, 3) tensor or ndarray — ground-truth global vertices
    soma_faces: (F, 3) int ndarray
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", pyopengl_platform)
    if egl_device_id is not None:
        os.environ.setdefault("EGL_DEVICE_ID", str(egl_device_id))
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        Log.warning(f"[MetricMetrosimBoth] pyrender/trimesh unavailable, skip mesh vis: {exc}")
        return

    from gem.utils.video_io_utils import get_writer
    from gem.utils.vis.renderer import get_global_cameras_static_v2, get_ground_params_from_points
    from gem.utils.vis.renderer_tools import checkerboard_geometry

    if torch.is_tensor(pred_verts):
        pred_verts = pred_verts.detach().cpu().numpy().astype(np.float32)
    if torch.is_tensor(gt_verts):
        gt_verts = gt_verts.detach().cpu().numpy().astype(np.float32)

    faces_int = soma_faces.astype(np.int32)
    L = min(len(pred_verts), vis_max_frames)
    pred_verts = pred_verts[:L]
    gt_verts = gt_verts[:L]

    # Camera and ground params derived from pred trajectory
    pred_t = torch.from_numpy(pred_verts)
    scale, cx, cz = get_ground_params_from_points(pred_t[:, 0], pred_t)
    position, target, up = get_global_cameras_static_v2(
        pred_t, beta=4.0, cam_height_degree=20, target_center_height=1.0
    )
    cam_pose = _look_at(position.cpu().numpy(), target.cpu().numpy(), up.cpu().numpy())

    # Checkerboard ground geometry (numpy, RGBA colors)
    gv, gf, gvc, _ = checkerboard_geometry(
        length=max(float(scale), 3.0) * 2, c1=float(cx), c2=float(cz), up="y"
    )
    gf_int = gf.astype(np.int32)

    focal = float(image_size) * 1.5

    # Ensure headless pyrender (mock pyglet/X11 if needed)
    import sys
    import types as _types

    if "pyrender.viewer" not in sys.modules:
        _viewer_mod = _types.ModuleType("pyrender.viewer")
        _viewer_mod.Viewer = type("Viewer", (), {})
        sys.modules["pyrender.viewer"] = _viewer_mod

    def _build_scene():
        scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.3, 0.3, 0.3))
        camera = pyrender.IntrinsicsCamera(
            fx=focal, fy=focal, cx=image_size / 2.0, cy=image_size / 2.0
        )
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)
        # Ground plane (static)
        ground_rgb = (gvc[:, :3] * 255).clip(0, 255).astype(np.uint8)
        ground_tm = trimesh.Trimesh(
            vertices=gv, faces=gf_int, vertex_colors=ground_rgb, process=False
        )
        ground_mesh = pyrender.Mesh.from_trimesh(ground_tm, smooth=False)
        scene.add(ground_mesh, name="ground")
        return scene

    def _add_body(scene, verts, color_rgba):
        normals = _compute_vertex_normals_np(verts, faces_int)
        vc = np.tile(np.array(color_rgba, dtype=np.float32), (len(verts), 1))
        mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.5, doubleSided=True
        )
        prim = pyrender.Primitive(
            positions=verts, normals=normals, indices=faces_int, color_0=vc, material=mat, mode=4
        )
        mesh = pyrender.Mesh(primitives=[prim])
        return scene.add(mesh, name="body")

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        scene_pred = _build_scene()
        scene_gt = _build_scene()
        r = pyrender.OffscreenRenderer(image_size, image_size)
        writer = get_writer(out_path, fps=fps)

        body_pred_node = None
        body_gt_node = None
        pred_color = (0.69, 0.39, 0.96, 1.0)  # purple — predicted
        gt_color = (0.27, 0.82, 0.51, 1.0)  # green — ground truth

        for t in range(L):
            if body_pred_node is not None:
                scene_pred.remove_node(body_pred_node)
            body_pred_node = _add_body(scene_pred, pred_verts[t], pred_color)

            if body_gt_node is not None:
                scene_gt.remove_node(body_gt_node)
            body_gt_node = _add_body(scene_gt, gt_verts[t], gt_color)

            frame_pred, _ = r.render(scene_pred)
            frame_gt, _ = r.render(scene_gt)
            frame = np.concatenate([frame_pred, frame_gt], axis=1)  # (H, 2W, 3)
            writer.write_frame(frame)

        writer.close()
        r.delete()
        Log.info(f"[MetricMetrosimBoth] Global mesh vis → {out_path}")
    except Exception as exc:
        Log.warning(f"[MetricMetrosimBoth] Global mesh vis failed: {exc}")


def _visualize_incam_mesh_overlay_mp4(
    pred_verts_c,
    soma_faces: np.ndarray,
    K_fullimg,
    out_path: str,
    fps: int = 30,
    vis_max_frames: int = 150,
):
    """Renders in-cam mesh overlay on a white background using Pyrender.

    pred_verts_c: (L, V, 3) tensor or ndarray — predicted in-cam vertices
    K_fullimg:    (L, 3, 3) or (3, 3) tensor or ndarray
    """
    try:
        from gem.utils.vis.pyrender_incam import Renderer
    except ImportError as exc:
        Log.warning(f"[MetricMetrosimBoth] pyrender/trimesh unavailable, skip incam vis: {exc}")
        return

    from gem.utils.video_io_utils import get_writer

    if torch.is_tensor(pred_verts_c):
        pred_verts_c = pred_verts_c.detach().cpu().numpy().astype(np.float32)
    if torch.is_tensor(K_fullimg):
        K_fullimg = K_fullimg.detach().cpu().numpy().astype(np.float32)
    if K_fullimg.ndim == 2:
        K_fullimg = np.broadcast_to(K_fullimg[None], (len(pred_verts_c), 3, 3))

    L = min(len(pred_verts_c), vis_max_frames)
    pred_verts_c = pred_verts_c[:L]

    focal = float(K_fullimg[0, 0, 0])
    cx, cy = float(K_fullimg[0, 0, 2]), float(K_fullimg[0, 1, 2])
    img_w = max(int(round(cx * 2)), 64)
    img_h = max(int(round(cy * 2)), 64)

    renderer = Renderer(focal_length=focal, faces=soma_faces.astype(np.int32))
    cam_t = np.zeros(3, dtype=np.float32)
    bg = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = get_writer(out_path, fps=fps)
        for t in range(L):
            out_img = renderer(
                pred_verts_c[t],
                cam_t,
                bg.copy(),
                camera_center=[cx, cy],
            )
            writer.write_frame(out_img)
        writer.close()
        Log.info(f"[MetricMetrosimBoth] Incam mesh vis → {out_path}")
    except Exception as exc:
        Log.warning(f"[MetricMetrosimBoth] Incam mesh vis failed: {exc}")


def _as_int(x):
    if isinstance(x, int | np.integer):
        return int(x)
    if torch.is_tensor(x):
        return int(x.item())
    return int(x)


def _trim_b1_seq(d, seq_length: int):
    """Input values may be (B, L, ...) or (L, ...). Returns (1, seq_length, ...)."""
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            if v.ndim >= 2 and v.shape[0] == 1:
                v = v[0]
            v = v[:seq_length]
            out[k] = v[None]
        else:
            out[k] = v
    return out


@torch.no_grad()
def _compute_global_metrics_no_fs(batch):
    pred_j3d_glob = batch["pred_j3d_glob"].cpu()
    target_j3d_glob = batch["target_j3d_glob"].cpu()

    seq_length = pred_j3d_glob.shape[0]
    chunk_length = 100
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)
        target_j3d = target_j3d_glob[start:end].clone()
        pred_j3d = pred_j3d_glob[start:end].clone()
        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)
        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm
    rte = compute_rte(target_j3d_glob[:, 0], pred_j3d_glob[:, 0]) * 1e2
    jitter = compute_jitter(pred_j3d_glob, fps=30)

    return {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "jitter": jitter,
        "fs": np.array([np.nan], dtype=np.float32),
    }


class MetricMetrosimBoth(pl.Callback):
    """Metrosim metric callback for SOMA body model.

    - In-cam metrics: compare outputs["pred_body_params_incam"] vs batch["soma_params_c"]
    - Global metrics: compare outputs["pred_body_params_global"] vs batch["soma_params_w"]

    Uses SOMA FK directly.
    Aggregates per meta["data_name"] so multiple Metrosim val sets report separately.
    """

    def __init__(
        self,
        vis_every_n_val: int = 5,
        vis_mode: str = "off",
        vis_max_frames: int = 150,
        vis_max_sequences: int = 5,
        vis_out_dir: str = "out",
        vis_pyopengl_platform: str = "osmesa",
        vis_egl_device_id=None,
        vis_mesh_backend: str = "auto",
    ):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.vis_mode = vis_mode
        self.vis_max_frames = vis_max_frames
        self.vis_max_sequences = vis_max_sequences
        self.vis_out_dir = vis_out_dir
        self.vis_pyopengl_platform = vis_pyopengl_platform
        self.vis_egl_device_id = vis_egl_device_id
        self.vis_mesh_backend = vis_mesh_backend
        self.num_val = 0
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
            "wa2_mpjpe": {},
            "waa_mpjpe": {},
            "rte": {},
            "jitter": {},
            "fs": {},
        }
        self.soma_model = SomaLayer(
            data_root="inputs/soma_assets",
            low_lod=True,
            device="cuda",
            identity_model_type="mhr",
            mode="warp",
        )
        if self.vis_mode != "off":
            _faces = self.soma_model.faces
            self.soma_faces = (
                _faces.detach().cpu().numpy() if torch.is_tensor(_faces) else np.asarray(_faces)
            )

        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wandb_vis_dict = {}
        self._num_vis_this_epoch = 0

    def _update_metric(self, metric_key: str, data_name: str, vid: str, value):
        self.metric_aggregator[metric_key].setdefault(data_name, {})[vid] = as_np_array(value)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(batch, dict) or "meta" not in batch:
            return

        meta0 = batch["meta"][0] if isinstance(batch.get("meta"), list | tuple) else batch["meta"]
        if meta0.get("dataset_id") != "metrosim":
            return

        if "soma_params_c" not in batch:
            Log.warning("[MetricMetrosimBoth] Missing soma_params_c in batch; skip.")
            return

        if torch.cuda.is_available():
            self.soma_model = self.soma_model.cuda()

        vid = meta0.get("vid", f"idx{batch_idx}")
        data_name = meta0.get("data_name", "metrosim")
        seq_length = _as_int(batch["length"][0])

        if "pred_body_params_incam" not in outputs:
            Log.warning(
                f"[MetricMetrosimBoth] Missing pred_body_params_incam for {data_name}/{vid}, skip."
            )
            return

        # ---- Ground truth ----
        gt_c_params = {k: v[0][:seq_length] for k, v in batch["soma_params_c"].items()}
        gt_w_params = {k: v[0][:seq_length] for k, v in batch["soma_params_w"].items()}

        gt_c_out = self.soma_model(**_trim_b1_seq(gt_c_params, seq_length))
        gt_w_out = self.soma_model(**_trim_b1_seq(gt_w_params, seq_length))

        # ---- Prediction (in-cam) ----
        pred_c_params = _trim_b1_seq(outputs["pred_body_params_incam"], seq_length)
        pred_c_out = self.soma_model(**pred_c_params)

        # ---- In-cam metrics ----
        batch_eval_cam = {
            "pred_j3d": pred_c_out["joints"][0],
            "target_j3d": gt_c_out["joints"][0],
            "pred_verts": pred_c_out["vertices"][0],
            "target_verts": gt_c_out["vertices"][0],
        }
        try:
            camcoord_metrics = compute_camcoord_metrics(batch_eval_cam, pelvis_idxs=[0])
        except Exception as e:
            Log.warning(f"[MetricMetrosimBoth] compute_camcoord_metrics failed for {vid}: {e}")
            return
        for k, v in camcoord_metrics.items():
            self._update_metric(k, data_name, vid, v)

        # ---- Global metrics ----
        if "pred_body_params_global" not in outputs:
            return

        T_w2ay = batch["T_w2ay"][0]  # (1, 4, 4) or (4, 4) after collation
        pred_w_params = _trim_b1_seq(outputs["pred_body_params_global"], seq_length)
        pred_w_out = self.soma_model(**pred_w_params)

        target_j3d_glob = apply_T_on_points(gt_w_out["joints"][0], T_w2ay)
        target_verts_glob = apply_T_on_points(gt_w_out["vertices"][0], T_w2ay)

        batch_eval_global = {
            "pred_j3d_glob": pred_w_out["joints"][0],
            "target_j3d_glob": target_j3d_glob,
            "pred_verts_glob": pred_w_out["vertices"][0],
            "target_verts_glob": target_verts_glob,
        }
        try:
            global_metrics = compute_global_metrics(batch_eval_global)
        except Exception:
            global_metrics = _compute_global_metrics_no_fs(batch_eval_global)
        for k, v in global_metrics.items():
            self._update_metric(k, data_name, vid, v)

        # ---- Mesh MP4 visualization (global, pred|gt) ----
        if (
            trainer.global_rank == 0
            and self.vis_mode != "off"
            and (self.num_val % max(self.vis_every_n_val, 1) == 0)
            and self._num_vis_this_epoch < self.vis_max_sequences
        ):
            wandb_vis = getattr(self, "wandb_vis_dict", {})

            # In-cam mesh overlay on white background (pred only)
            if "K_fullimg" in batch:
                incam_out = (
                    Path(self.vis_out_dir)
                    / f"vis_metrosim_mesh_{data_name}_incam"
                    / f"{batch_idx}.mp4"
                )
                _visualize_incam_mesh_overlay_mp4(
                    pred_c_out["vertices"][0],
                    self.soma_faces,
                    batch["K_fullimg"][0],
                    str(incam_out),
                    vis_max_frames=self.vis_max_frames,
                )
                try:
                    import wandb

                    from gem.utils.tools import wandb_run_exists

                    if wandb_run_exists() and incam_out.exists():
                        wandb_vis[f"vis_METROSIM_incam/{data_name}/{batch_idx}"] = wandb.Video(
                            str(incam_out)
                        )
                except Exception:
                    pass

            # Global pred|gt mesh MP4
            backend = (self.vis_mesh_backend or "auto").lower()
            if backend in ("pyrender", "auto"):
                pred_v = pred_w_out["vertices"][0]
                gt_v = batch_eval_global["target_verts_glob"]
                video_out = (
                    Path(self.vis_out_dir)
                    / f"vis_metrosim_mesh_{data_name}_video"
                    / f"{batch_idx}.mp4"
                )
                _visualize_mesh_pyrender_mp4(
                    pred_v,
                    gt_v,
                    self.soma_faces,
                    str(video_out),
                    vis_max_frames=self.vis_max_frames,
                    pyopengl_platform=self.vis_pyopengl_platform,
                    egl_device_id=self.vis_egl_device_id,
                )
                try:
                    import wandb

                    from gem.utils.tools import wandb_run_exists

                    if wandb_run_exists() and video_out.exists():
                        wandb_vis[f"vis_METROSIM_global/{data_name}/{batch_idx}"] = wandb.Video(
                            str(video_out)
                        )
                except Exception:
                    pass

            self._num_vis_this_epoch += 1

    def on_predict_epoch_end(self, trainer, pl_module):
        self.num_val += 1
        local_rank, world_size = trainer.local_rank, trainer.world_size
        metric_keys = list(self.metric_aggregator.keys())

        with torch.inference_mode(False):
            metric_aggregator_gathered = all_gather(self.metric_aggregator)

        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                for data_name, per_vid in d[metric_key].items():
                    self.metric_aggregator[metric_key].setdefault(data_name, {}).update(per_vid)

        if local_rank == 0:
            total = sum(len(v) for v in self.metric_aggregator.get("mpjpe", {}).values())
            Log.info(f"[MetricMetrosimBoth] {total} sequences evaluated across {world_size} ranks.")

        if pl_module.logger is not None and local_rank == 0:
            cur_step = trainer.global_step
            all_data_names = sorted(
                set().union(*[set(v.keys()) for v in self.metric_aggregator.values()])
            )
            for data_name in all_data_names:
                metrics_avg = {}
                for k in metric_keys:
                    vals = list(self.metric_aggregator[k].get(data_name, {}).values())
                    if not vals:
                        continue
                    try:
                        metrics_avg[k] = float(np.nanmean(np.concatenate(vals)))
                    except Exception:
                        metrics_avg[k] = float(np.nanmean([np.nanmean(x) for x in vals]))

                Log.info(
                    f"[Metrics] METROSIM/{data_name}:\n"
                    + "\n".join(f"{k}: {v:.3f}" for k, v in metrics_avg.items())
                    + "\n------"
                )
                for k, v in metrics_avg.items():
                    pl_module.logger.log_metrics(
                        {f"val_metric_METROSIM/{data_name}/{k}": v}, step=cur_step
                    )

        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}

        # Flush accumulated wandb visualization entries
        if local_rank == 0 and pl_module.logger is not None:
            wandb_vis = getattr(self, "wandb_vis_dict", {})
            if wandb_vis:
                pl_module.logger.log_metrics(wandb_vis)
                self.wandb_vis_dict = {}
