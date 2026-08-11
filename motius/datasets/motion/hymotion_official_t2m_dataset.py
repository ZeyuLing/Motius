"""Official HY-Motion T2M record-file dataset.

This dataset mirrors the original HYMotion ``T2M_O6D_Plus_Dataset`` data
entry contract while reusing motius transforms.  It reads the official
``_input_record_files/*`` JSONs directly so record weighting, duplicate motion
entries across text sources, and ``data_src`` are preserved.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mmengine.dataset import BaseDataset
from mmengine import print_log
from tqdm import tqdm

from motius.datasets.motion.motionhub.flexible_collate import flexible_collate
from motius.registry import DATASETS


def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _caption_dir_from_text_emb_dir(text_emb_dir: str) -> str:
    """Return the caption dir used only to derive the sibling qwen3 .pt path."""
    if text_emb_dir == "qwen3_human_checked_short":
        return "human_checked_caption"
    if text_emb_dir == "qwen3_improved_simple_short":
        return "improved_simple_caption"
    if text_emb_dir.startswith("qwen3_") and text_emb_dir.endswith("_short"):
        return text_emb_dir[len("qwen3_") : -len("_short")] + "_caption"
    if text_emb_dir.startswith("qwen3_"):
        return text_emb_dir[len("qwen3_") :] + "_caption"
    return text_emb_dir + "_caption"


@DATASETS.register_module(force=True)
class HYMotionOfficialT2MDataset(BaseDataset):
    """Read official HY-Motion T2M input-record files.

    Args:
        data_root: HYMotion data root containing ``Academic/``, ``Taobao/``,
            and ``_input_record_files/``.
        input_record_file_dir: Directory or explicit list of official record
            JSONs.  Relative paths are resolved under ``data_root``.
        motion_dir: Directory name inside each source root containing O6DP npy.
        motion_postfix: Motion file suffix without dot.
        pipeline: motius transform pipeline.
        require_motion_file: If true, skip records whose O6DP motion file is
            absent during dataset initialization.
        max_records: Optional debug limit.
    """

    collate_fn = staticmethod(flexible_collate)

    def __init__(
        self,
        data_root: str = "data/hymotion_data",
        input_record_file_dir: str | Sequence[str] = "_input_record_files/sft_train_v1103_qwen3",
        motion_dir: str = "motions_o6dp_v0922",
        motion_postfix: str = "npy",
        pipeline: Optional[List[Dict[str, Any]]] = None,
        require_motion_file: bool = False,
        max_records: Optional[int] = None,
        refetch: bool = True,
        max_refetch: int = 100,
        verbose: bool = True,
    ):
        self.data_root = _resolve_path(data_root)
        self.input_record_file_dir = input_record_file_dir
        self.motion_dir = motion_dir
        self.motion_postfix = motion_postfix.lstrip(".")
        self.require_motion_file = bool(require_motion_file)
        self.max_records = max_records
        self.refetch = bool(refetch)
        self.max_refetch = int(max_refetch)
        self.verbose = verbose
        super().__init__(
            ann_file="",
            metainfo=None,
            data_root=self.data_root,
            data_prefix={},
            serialize_data=False,
            pipeline=list(pipeline) if pipeline is not None else [],
            test_mode=False,
            lazy_init=False,
        )

    def _record_files(self) -> List[Path]:
        paths = self.input_record_file_dir
        if isinstance(paths, (str, os.PathLike)):
            path = Path(paths)
            if not path.is_absolute():
                path = Path(self.data_root) / path
            if path.is_dir():
                return sorted(p for p in path.iterdir() if p.suffix == ".json")
            if path.is_file():
                return [path]
            raise FileNotFoundError(f"input_record_file_dir does not exist: {path}")

        result = []
        for item in paths:
            path = Path(item)
            if not path.is_absolute():
                path = Path(self.data_root) / path
            if not path.is_file():
                raise FileNotFoundError(f"input record file does not exist: {path}")
            result.append(path)
        return sorted(result)

    def load_data_list(self) -> List[dict]:
        record_files = self._record_files()
        data_list: List[dict] = []
        skipped_missing_motion = 0
        total_records = 0
        iterator = tqdm(record_files, desc="Loading HYMotion SFT records") if self.verbose else record_files

        for record_file in iterator:
            with open(record_file, "r") as f:
                record = json.load(f)
            root_dir = str(record["root_dir"]).strip("/")
            text_emb_dir = str(record["text_emb_dir"])
            file_list = record.get("file_list", [])
            input_dir = Path(self.data_root) / root_dir
            caption_dir = _caption_dir_from_text_emb_dir(text_emb_dir)
            data_src = root_dir.split("/", 1)[0]

            for file_id in file_list:
                total_records += 1
                motion_path = input_dir / self.motion_dir / f"{file_id}.{self.motion_postfix}"
                if self.require_motion_file and not motion_path.exists():
                    skipped_missing_motion += 1
                    continue
                caption_path = input_dir / caption_dir / f"{file_id}.json"
                data_list.append(
                    {
                        "motion_path": str(motion_path),
                        "caption_path": str(caption_path),
                        "data_src": data_src,
                        "source": root_dir,
                        "root_dir": root_dir,
                        "text_emb_dir": text_emb_dir,
                        "input_filename": file_id,
                        "subset": data_src.lower(),
                        "fps": 30.0,
                        "has_hand": False,
                        "duration": 0.0,
                        "num_frames": 0,
                    }
                )
                if self.max_records is not None and len(data_list) >= int(self.max_records):
                    break
            if self.max_records is not None and len(data_list) >= int(self.max_records):
                break

        print_log(
            "Loaded "
            f"{len(data_list)} HYMotion official SFT records from {len(record_files)} files "
            f"(total_records={total_records}, skipped_missing_motion={skipped_missing_motion})"
        )
        return data_list

    def prepare_data(self, idx: int) -> dict:
        return dict(self.data_list[idx])

    def __getitem__(self, idx: int) -> dict:
        """Run the pipeline and resample when a transform rejects the record."""
        last_error: Optional[Exception] = None
        for _ in range(self.max_refetch + 1):
            try:
                data = self.prepare_data(idx)
                data = self.pipeline(data)
                if data is not None:
                    return data
            except Exception as exc:
                last_error = exc
                if not self.refetch:
                    raise
            if not self.refetch:
                break
            idx = random.randrange(len(self.data_list))
        if last_error is not None:
            raise RuntimeError(
                f"Failed to fetch a valid HYMotion official T2M sample after "
                f"{self.max_refetch} retries"
            ) from last_error
        raise RuntimeError(
            f"Failed to fetch a valid HYMotion official T2M sample after "
            f"{self.max_refetch} retries"
        )
