from typing import Dict, List, Sequence, Union, Tuple
from mmcv import BaseTransform
import numpy as np
import torch
from motius.motion.representation.rotation import (
    ROTATION_TYPE,
    axis_angle_to_euler,
    axis_angle_to_quaternion,
    axis_angle_to_rotation_6d,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from motius.registry import TRANSFORMS


def process_smplx_pose(
    pose_55_axis_angle: np.ndarray,  # [T, 165] or [T, 55, 3]
    rot_type: str,
    out_type: str,
    rot6d_convention: str = "row",
) -> np.ndarray:
    """
    Convert SMPL-X 55-joint axis-angle pose to target joint set & rotation representation.

    Args:
        pose_55_axis_angle: [T, 165] or [T, 55, 3], axis-angle in radians.
        rot_type: "axis_angle" | "rotation_6d" | "quaternion" | "euler"
        out_type: "smpl_22" | "smplh" | "smplx_55"
        rot6d_convention: "row" (default, backward-compat) or "column".
            Only affects rot_type="rotation_6d".
            - "row": output is row-major [R00,R01, R10,R11, R20,R21] (HyMotion M2M convention)
            - "column": output is column-major [R00,R10,R20, R01,R11,R21] (PRISM/VerMo convention)

    Returns:
        pose: [T, J * D], where D=3 (axis_angle/euler), 4 (quaternion), 6 (rotation_6d)
    """
    assert out_type in ["smpl_22", "smplh", "smplx_55"]
    assert rot_type in ["axis_angle", "rotation_6d", "quaternion", "euler"]

    # reshape to [T, 55, 3]
    # SMPL-X 55 joints
    if pose_55_axis_angle.ndim == 2 and pose_55_axis_angle.shape[1] == 55 * 3:
        T = pose_55_axis_angle.shape[0]
        aa = pose_55_axis_angle.reshape(T, 55, 3)
    elif pose_55_axis_angle.ndim == 3 and pose_55_axis_angle.shape[1:] == (55, 3):
        T = pose_55_axis_angle.shape[0]
        aa = pose_55_axis_angle
    # SMPL-H 52 joints, padding jaw/eyes to 55 joints
    elif (
        pose_55_axis_angle.ndim == 2 and pose_55_axis_angle.shape[1] == 52 * 3
    ):  # SMPL-H
        T = pose_55_axis_angle.shape[0]
        pose_55_axis_angle = np.concatenate(
            [pose_55_axis_angle[:, :66], np.zeros((T, 9)), pose_55_axis_angle[:, 66:]],
            axis=1,
        )
        aa = pose_55_axis_angle.reshape(T, 55, 3)
    elif pose_55_axis_angle.ndim == 3 and pose_55_axis_angle.shape[1:] == (52, 3):
        T = pose_55_axis_angle.shape[0]
        aa = np.concatenate(
            [
                pose_55_axis_angle[:, :22],
                np.zeros((T, 3, 3)),
                pose_55_axis_angle[:, 22:],
            ],
            axis=1,
        )
    else:
        raise ValueError(
            f"pose_55_axis_angle must be [T,165] or [T,55,3] or [T,52,3], got {pose_55_axis_angle.shape}"
        )

    # Joint subsets (SMPL-X-55 ordering)
    IDX_SMPL22 = np.arange(22, dtype=np.int64)  # 0..21
    IDX_SMPLH = np.arange(52, dtype=np.int64)  # 0..51
    IDX_SMPLX55 = np.arange(55, dtype=np.int64)  # 0..54

    if out_type == "smpl_22":
        sel = IDX_SMPL22
    elif out_type == "smplh":
        sel = IDX_SMPLH
    else:
        sel = IDX_SMPLX55

    aa = aa[:, sel, :]  # [T, J, 3]
    T, J, _ = aa.shape
    aa_flat = aa.reshape(T * J, 3)  # [T*J, 3]

    # Convert rotation representation
    if rot_type == "axis_angle":
        out = aa  # [T,J,3]
        D = 3
    elif rot_type == "rotation_6d":
        out = axis_angle_to_rotation_6d(aa_flat).reshape(T, J, 6)
        # axis_angle_to_rotation_6d outputs column-major: [R00,R10,R20, R01,R11,R21]
        if rot6d_convention == "row":
            # HyMotion convention is row-major: [R00,R01, R10,R11, R20,R21]
            # Rearrange: col_major[0,3,1,4,2,5] -> row_major
            out = out[:, :, [0, 3, 1, 4, 2, 5]]
        # else "column": keep as-is (PRISM/VerMo convention)
        D = 6
    elif rot_type == "quaternion":
        out = axis_angle_to_quaternion(aa_flat).reshape(T, J, 4)
        D = 4
    elif rot_type == "euler":
        out = axis_angle_to_euler(aa_flat).reshape(T, J, 3)
        D = 3
    else:
        raise ValueError(f"Unknown rot_type: {rot_type}")

    return out.reshape(T, J * D).astype(np.float32)


def process_transl(abs_trans: np.ndarray, transl_type: str) -> np.ndarray:
    """
    Process absolute translation to relative or absolute-relative representation.

    Args:
        abs_trans: [T, 3], absolute translation (world coords).
        transl_type: "abs" | "rel" | "abs_rel"

    Returns:
        transl: [T, 3] for "abs"/"rel"; [T, 6] for "abs_rel".
    """
    if transl_type == "abs":
        return abs_trans
    elif transl_type == "rel":
        return np.concatenate(
            [np.zeros((1, 3), dtype=abs_trans.dtype), abs_trans[1:] - abs_trans[:-1]],
            axis=0,
        )
    elif transl_type == "abs_rel":
        rel_transl = np.concatenate(
            [np.zeros((1, 3), dtype=abs_trans.dtype), abs_trans[1:] - abs_trans[:-1]],
            axis=0,
        )
        abs_rel_transl = np.concatenate([abs_trans, rel_transl], axis=-1)
        return abs_rel_transl
    else:
        raise ValueError(f"Unknown transl_type: {transl_type}")


def _build_Ry_from_deg(deg: float) -> np.ndarray:
    """Y-up（Y 为向上轴）的绕 Y 轴旋转矩阵，角度制输入。"""
    yaw = np.deg2rad(deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R_y = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )
    return R_y


def apply_root_yaw_to_axis_angle(
    pose_55_axis_angle: np.ndarray,
    R_y: np.ndarray,
) -> np.ndarray:
    """
    用给定的世界系 Y 轴旋转 R_y，对整段序列的根关节(索引0)做左乘：
        R_root'(t) = R_y · R_root(t)
    其余关节不变；输入/输出保持 axis-angle 形式与原形状一致。
    """
    # 归一到 [T,55,3]
    if pose_55_axis_angle.ndim == 2 and pose_55_axis_angle.shape[1] == 55 * 3:
        T = pose_55_axis_angle.shape[0]
        aa = pose_55_axis_angle.reshape(T, 55, 3).copy()
        flatten_back = True
    elif pose_55_axis_angle.ndim == 3 and pose_55_axis_angle.shape[1:] == (55, 3):
        aa = pose_55_axis_angle.copy()
        T = aa.shape[0]
        flatten_back = False
    elif (
        pose_55_axis_angle.ndim == 2 and pose_55_axis_angle.shape[1] == 52 * 3
    ):  # SMPL-H
        T = pose_55_axis_angle.shape[0]
        pose_55_axis_angle = np.concatenate(
            [pose_55_axis_angle[:, :66], np.zeros((T, 9)), pose_55_axis_angle[:, 66:]],
            axis=1,
        )
        aa = pose_55_axis_angle.reshape(T, 55, 3)
        flatten_back = True
    elif pose_55_axis_angle.ndim == 3 and pose_55_axis_angle.shape[1:] == (52, 3):
        T = pose_55_axis_angle.shape[0]
        aa = np.concatenate(
            [
                pose_55_axis_angle[:, :22],
                np.zeros((T, 3, 3)),
                pose_55_axis_angle[:, 22:],
            ],
            axis=1,
        )
        flatten_back = False
    else:
        raise ValueError(
            f"pose_55_axis_angle must be [T,165] or [T,55,3] or [T,52,3], got {pose_55_axis_angle.shape}"
        )

    # 根关节索引=0（SMPL/SMPL-X 约定）
    root_aa = aa[:, 0, :]  # [T,3]
    root_R = axis_angle_to_matrix(root_aa.reshape(-1, 3))  # [T,3,3]
    R_y_batched = np.broadcast_to(R_y[None, :, :], root_R.shape).astype(np.float32)
    root_R_new = np.matmul(R_y_batched, root_R)  # [T,3,3]
    root_aa_new = matrix_to_axis_angle(root_R_new).reshape(T, 3)
    aa[:, 0, :] = root_aa_new

    if flatten_back:
        return aa.reshape(T, 55 * 3).astype(np.float32)
    return aa.astype(np.float32)


def _read_one_person_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Union[int, None]]:
    """读取单人 npz：返回 (trans[T,3], poses[T,165], fps or None)"""
    data = np.load(path, allow_pickle=True)
    abs_trans = np.asarray(data["trans"], dtype=np.float32)  # world root translation
    poses = np.asarray(data["poses"], dtype=np.float32)  # axis-angle packed [T, 55*3]
    fps = int(data["mocap_framerate"]) if "mocap_framerate" in data else None
    return abs_trans, poses, fps


@TRANSFORMS.register_module(force=True)
class LoadSmplx55(BaseTransform):
    """
    同时支持单人/多人 SMPL-X 读取与 Y-up 刚体增强（群体一致）。
    - 单人：results[f"{key}_path"] 是 str，返回 [T, D] 的 torch.FloatTensor。
    - 多人：results[f"{key}_path"] 是 List[str]，返回 [P, T, D] 的 torch.FloatTensor （群体使用同一 R_y 与 XZ 偏移）。

    增强构成：以概率 p 采样 yaw∈[-deg, +deg] 绕 Y 轴旋转 + XZ 平面全局偏移（y=0）。
    对每个参与者：abs translation 与 root global_orient 一起被变换；body pose（非根）不变。
    """

    def __init__(
        self,
        key: str = "motion",
        rot_type: str = "rotation_6d",
        transl_type: str = "abs",
        smpl_type: str = "smpl_22",
        rot6d_convention: str = "row",
        # ===== Augmentation knobs (Y-up world) =====
        # Default OFF: augmentation changes data distribution (especially root rotation
        # and translation), so mean/std stats must be computed with matching aug settings.
        # Enable explicitly in config when needed.
        transl_aug_prob: float = 0.0,  # 概率做增强（同时作用于 translation 与 root 朝向）
        transl_aug_yaw_deg: float = 180.0,  # 绕 Y 轴随机旋转角度范围 [-deg, +deg]
        transl_aug_offset_std: Tuple[float, float, float] = (
            1.0,
            0.0,
            1.0,
        ),  # 仅 XZ 平面偏移；y=0
        # ===== Multi-person consistency =====
        require_same_T: bool = True,  # 多人是否要求相同帧数 T
        require_same_fps: bool = True,  # 多人是否要求相同 fps
    ):
        super().__init__()
        assert smpl_type in ["smpl_22", "smplh", "smplx_55"]
        assert rot_type in [
            ROTATION_TYPE.AXIS_ANGLE,
            ROTATION_TYPE.ROTATION_6D,
            ROTATION_TYPE.QUATERNION,
            ROTATION_TYPE.EULER,
        ]
        assert transl_type in ["abs", "rel", "abs_rel"]
        assert rot6d_convention in ("row", "column"), (
            f"rot6d_convention must be 'row' or 'column', got '{rot6d_convention}'"
        )

        self.key = key
        self.rot_type = rot_type
        self.smpl_type = smpl_type
        self.transl_type = transl_type
        self.rot6d_convention = rot6d_convention

        # augmentation params
        self.transl_aug_prob = float(transl_aug_prob)
        self.transl_aug_yaw_deg = float(transl_aug_yaw_deg)
        self.transl_aug_offset_std = np.asarray(transl_aug_offset_std, dtype=np.float32)

        # multi-person constraints
        self.require_same_T = bool(require_same_T)
        self.require_same_fps = bool(require_same_fps)

    @staticmethod
    def _has_crop_metadata(results: Dict) -> bool:
        return any(
            results.get(key) is not None
            for key in (
                "_motion_audio_crop_start_frame",
                "_motion_audio_crop_start",
                "_motion_audio_crop_num_frames",
                "_motion_audio_crop_duration",
            )
        )

    @staticmethod
    def _crop_bounds_from_results(results: Dict, fps: Union[int, float, None], total_frames: int) -> Tuple[int, int]:
        """Return deterministic crop bounds carried by annotation metadata.

        The keys are shared with ``LoadAudio``/``MotionAudioRandomCrop`` so an
        overfit annotation can crop motion and audio/music to the same segment
        before tokenization.  Datasets without these keys keep the full clip.
        """
        if total_frames <= 0:
            return 0, 0
        if fps is None:
            fps = 0
        fps = float(fps)

        def has_value(key: str) -> bool:
            return results.get(key) is not None

        has_crop = LoadSmplx55._has_crop_metadata(results)
        results["_motion_audio_crop_metadata_explicit"] = bool(has_crop)
        if not has_crop:
            return 0, int(total_frames)

        if has_value("_motion_audio_crop_start_frame"):
            start = int(round(float(results["_motion_audio_crop_start_frame"])))
        elif fps > 0 and has_value("_motion_audio_crop_start"):
            start = int(round(float(results["_motion_audio_crop_start"]) * fps))
        else:
            start = 0

        if has_value("_motion_audio_crop_num_frames"):
            num_frames = int(round(float(results["_motion_audio_crop_num_frames"])))
        elif fps > 0 and has_value("_motion_audio_crop_duration"):
            num_frames = int(round(float(results["_motion_audio_crop_duration"]) * fps))
        else:
            num_frames = int(total_frames) - start

        start = max(0, min(start, int(total_frames) - 1))
        num_frames = max(1, min(num_frames, int(total_frames) - start))
        return start, num_frames

    @staticmethod
    def _write_crop_metadata(results: Dict, fps: Union[int, float, None], start: int, num_frames: int) -> None:
        if fps is None:
            return
        fps = float(fps)
        if fps <= 0:
            return
        results["_motion_audio_crop_start_frame"] = int(start)
        results["_motion_audio_crop_num_frames"] = int(num_frames)
        results["_motion_audio_crop_start"] = float(start / fps)
        results["_motion_audio_crop_duration"] = float(num_frames / fps)

    def _sample_group_transform(self) -> Tuple[bool, float, np.ndarray, np.ndarray]:
        """采样群体统一的增强：是否增强、yaw(deg)、R_y[3,3]、offset[3]（y=0）。"""
        do_aug = (self.transl_aug_prob > 0.0) and (
            np.random.rand() < self.transl_aug_prob
        )
        yaw_deg = 0.0
        R_y = np.eye(3, dtype=np.float32)
        offset = np.zeros(3, dtype=np.float32)
        if do_aug:
            yaw_deg = float(
                np.random.uniform(-self.transl_aug_yaw_deg, +self.transl_aug_yaw_deg)
            )
            R_y = _build_Ry_from_deg(yaw_deg)  # [3,3]
            sx, sy, sz = self.transl_aug_offset_std
            # 仅 XZ 平面偏移；强制 y=0
            offset = np.array(
                [
                    np.random.normal(0.0, float(sx)),
                    0.0,
                    np.random.normal(0.0, float(sz)),
                ],
                dtype=np.float32,
            )
        return do_aug, yaw_deg, R_y, offset

    def _process_one_person(
        self,
        abs_trans: np.ndarray,  # [T,3]
        poses_axis_angle: np.ndarray,  # [T,165]
        do_aug: bool,
        R_y: np.ndarray,  # [3,3]
        offset: np.ndarray,  # [3]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对单人的 (trans, poses) 应用群体共享的增强并转换表示。"""
        # === Translation path: 先增强再表示变换 ===
        if do_aug:
            abs_trans = abs_trans @ R_y.T
            abs_trans = abs_trans + offset[None, :]
        transl = process_transl(abs_trans, self.transl_type)  # [T,3] or [T,6]

        # === Pose path: 只对根关节左乘同一 R_y ===
        if do_aug:
            poses_axis_angle = apply_root_yaw_to_axis_angle(poses_axis_angle, R_y)
        pose = process_smplx_pose(
            poses_axis_angle, self.rot_type, self.smpl_type,
            rot6d_convention=self.rot6d_convention,
        )  # [T, J*D]

        return transl, pose  # numpy

    def _load_precomputed_135(self, data, results: Dict) -> Dict:
        """Fast path for NPZ files with pre-computed motion_135 (e.g., PerMo).

        These NPZs contain ``motion_135`` (T, 135) already in the target format
        [trans(3), rot6d_22joints(132)] but lack raw SMPL ``poses``/``trans``.
        No augmentation is applied (pre-computed data is used as-is).

        Parameters
        ----------
        data : NpzFile
            Already-loaded NPZ data (avoids double I/O).
        results : dict
            Pipeline results dict to update.
        """
        motion = np.asarray(data["motion_135"], dtype=np.float32)  # [T, 135]
        fps = int(data["mocap_framerate"]) if "mocap_framerate" in data else None
        start, T = self._crop_bounds_from_results(results, fps, motion.shape[0])
        motion = motion[start : start + T]
        self._write_crop_metadata(results, fps, start, T)

        results[self.key] = torch.from_numpy(motion)  # [T, 135]
        results["num_person"] = 1
        results["rot_type"] = self.rot_type
        results["smpl_type"] = self.smpl_type
        results["transl_type"] = self.transl_type
        results["rot6d_convention"] = self.rot6d_convention
        results["num_frames"] = T
        results["duration"] = (T / fps) if fps else None
        results["fps"] = fps
        results["num_joints"] = 22
        results["aug_yaw_deg"] = 0.0
        results["aug_offset"] = [0.0, 0.0, 0.0]
        return results

    def transform(self, results: Dict) -> Dict:
        path_or_list = results[f"{self.key}_path"]

        # ===================== 单人 =====================
        if isinstance(
            path_or_list,
            (
                str,
                np.str_,
            ),
        ):
            path = str(path_or_list)

            # Fast path: pre-computed motion_135 (e.g., PerMo dataset)
            # These NPZs have motion_135 [T, 135] but no raw SMPL poses/trans.
            data = np.load(path, allow_pickle=True)
            if "motion_135" in data and "poses" not in data:
                return self._load_precomputed_135(data, results)

            # Normal path: raw SMPL params
            abs_trans = np.asarray(data["trans"], dtype=np.float32)
            poses = np.asarray(data["poses"], dtype=np.float32)
            fps = int(data["mocap_framerate"]) if "mocap_framerate" in data else None
            start, crop_frames = self._crop_bounds_from_results(results, fps, abs_trans.shape[0])
            abs_trans = abs_trans[start : start + crop_frames]
            poses = poses[start : start + crop_frames]
            self._write_crop_metadata(results, fps, start, crop_frames)

            do_aug, yaw_deg, R_y, offset = self._sample_group_transform()
            transl, pose = self._process_one_person(
                abs_trans, poses, do_aug, R_y, offset
            )

            out = torch.from_numpy(np.concatenate([transl, pose], axis=-1))
            if torch.any(torch.isnan(out)):
                raise ValueError(
                    f"NaN values found in {path_or_list} after augmentation & concat."
                )

            results[self.key] = out  # [T, D]
            results["num_person"] = 1
            results["rot_type"] = self.rot_type
            results["smpl_type"] = self.smpl_type
            results["transl_type"] = self.transl_type
            results["rot6d_convention"] = self.rot6d_convention
            results["num_frames"] = int(pose.shape[0])
            results["duration"] = (results["num_frames"] / fps) if fps else None
            results["fps"] = fps
            results["num_joints"] = int(
                pose.shape[1]
                // (
                    6
                    if self.rot_type == "rotation_6d"
                    else 4 if self.rot_type == "quaternion" else 3
                )
            )
            results["aug_yaw_deg"] = yaw_deg if do_aug else 0.0
            results["aug_offset"] = offset.tolist() if do_aug else [0.0, 0.0, 0.0]
            return results

        # ===================== 多人 =====================
        if not isinstance(path_or_list, (list, tuple)):
            raise TypeError(
                f"{self.__class__.__name__}: expected str or List[str] at results['{self.key}_path'], "
                f"got {type(path_or_list)}"
            )

        paths: List[str] = [str(p) for p in path_or_list]
        persons: List[Tuple[np.ndarray, np.ndarray, Union[int, None]]] = [
            _read_one_person_npz(p) for p in paths
        ]

        # 基本一致性校验。固定 crop 的伪多人样本允许原始长度不同，
        # 但 crop 后仍必须对齐。
        T_list = [arr[0].shape[0] for arr in persons]  # 每人的 T
        fps_list = [arr[2] for arr in persons]
        has_crop = self._has_crop_metadata(results)
        if self.require_same_T and len(set(T_list)) != 1 and not has_crop:
            raise ValueError(
                f"{self.__class__.__name__}: all persons must have the same T when require_same_T=True, "
                f"got {T_list} for paths={paths}"
            )
        if self.require_same_fps and len(set(fps_list)) != 1:
            raise ValueError(
                f"{self.__class__.__name__}: all persons must have the same fps when require_same_fps=True, "
                f"got {fps_list} for paths={paths}"
            )

        # 采样一组群体共享的增强
        do_aug, yaw_deg, R_y, offset = self._sample_group_transform()

        crop_start, crop_frames = self._crop_bounds_from_results(
            results,
            fps_list[0] if fps_list else None,
            min(T_list) if T_list else 0,
        )
        if crop_frames > 0:
            persons = [
                (
                    abs_trans[crop_start : crop_start + crop_frames],
                    poses_axis_angle[crop_start : crop_start + crop_frames],
                    _fps,
                )
                for abs_trans, poses_axis_angle, _fps in persons
            ]
            self._write_crop_metadata(
                results,
                fps_list[0] if fps_list else None,
                crop_start,
                crop_frames,
                )

        T_after_crop = [arr[0].shape[0] for arr in persons]
        if self.require_same_T and len(set(T_after_crop)) != 1:
            raise ValueError(
                f"{self.__class__.__name__}: all persons must have the same T after crop when "
                f"require_same_T=True, got {T_after_crop} for paths={paths}"
            )

        # 逐人处理并堆叠
        transl_list, pose_list = [], []
        for abs_trans, poses_axis_angle, _fps in persons:
            transl_np, pose_np = self._process_one_person(
                abs_trans, poses_axis_angle, do_aug, R_y, offset
            )
            transl_list.append(transl_np)
            pose_list.append(pose_np)

        # 对齐 T（如果 require_same_T=False，可以截断到最小 T）
        if len(set([t.shape[0] for t in pose_list])) != 1:
            if self.require_same_T:
                # 按理不会到这里；双保险
                raise ValueError("Inconsistent T after processing.")
            else:
                min_T = min([t.shape[0] for t in pose_list])
                transl_list = [t[:min_T] for t in transl_list]
                pose_list = [p[:min_T] for p in pose_list]

        # 拼接每个人的 [T, D]
        person_feats = [
            np.concatenate([t, p], axis=-1) for t, p in zip(transl_list, pose_list)
        ]  # list of [T, D]
        stacked = torch.from_numpy(np.stack(person_feats, axis=0))  # [P, T, D]

        if torch.any(torch.isnan(stacked)):
            raise ValueError(
                f"NaN values found in multi-person augmentation & concat for paths={paths}"
            )

        # 写回 results
        results[self.key] = stacked  # [P, T, D]
        results["num_person"] = len(paths)
        results["rot_type"] = self.rot_type
        results["smpl_type"] = self.smpl_type
        results["transl_type"] = self.transl_type
        results["rot6d_convention"] = self.rot6d_convention
        results["num_frames"] = int(stacked.shape[1])
        # fps：若不强制一致，则回填列表；否则为公共 fps
        results["fps"] = fps_list if not self.require_same_fps else fps_list[0]
        results["duration"] = (
            [results["num_frames"] / f if f else None for f in fps_list]
            if not self.require_same_fps
            else (results["num_frames"] / fps_list[0] if fps_list[0] else None)
        )
        # 关节数（与单人相同）
        pose_dim = pose_list[0].shape[1]
        results["num_joints"] = int(
            pose_dim
            // (
                6
                if self.rot_type == "rotation_6d"
                else 4 if self.rot_type == "quaternion" else 3
            )
        )
        # 记录这次群体增强
        results["aug_yaw_deg"] = yaw_deg if do_aug else 0.0
        results["aug_offset"] = offset.tolist() if do_aug else [0.0, 0.0, 0.0]
        results[f"{self.key}_paths"] = paths
        return results
