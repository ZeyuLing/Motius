from collections import defaultdict
import hashlib
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Tuple, Type, Union
from tqdm import tqdm

try:
    from overrides import override
except ImportError:
    def override(method=None, **_kwargs):
        if method is None:
            return lambda fn: fn
        return method
import mmengine
from motius.models.vermo.task_utils import (
    ABBR_TASK_MAPPING,
    ALL_TASKS,
    abbr_list_to_task_list,
)
from motius.models.vermo.task_utils.modality import Audio, Caption, Modality
from motius.models.vermo.task_utils.task_lib.base_task import BaseTask
from motius.models.vermo.task_utils.task_lib.completion_tasks.motion_inbetween import MotionInbetween
from motius.datasets.motion.motionhub.single_agent_dataset import MotionHubSingleAgentDataset
from motius.datasets.motion.motionhub.flexible_collate import flexible_collate
from mmcv.transforms import BaseTransform
from mmengine.logging import print_log
from motius.registry import DATASETS


TASK_BUCKETS = (
    "caption_audio",
    "caption_non_audio",
    "audio_non_caption",
    "motion_non_caption",
)


def task_has_modal(task: Type[BaseTask], modal_cls: Type[Modality]) -> bool:
    try:
        for modal in task.all_modality():
            if isinstance(modal, type) and issubclass(modal, modal_cls):
                return True
            if isinstance(modal, modal_cls):
                return True
    except Exception:
        return False
    return False


def task_bucket_name(task: Type[BaseTask]) -> str:
    has_caption = task_has_modal(task, Caption)
    has_audio = task_has_modal(task, Audio)
    if has_caption and has_audio:
        return "caption_audio"
    if has_caption:
        return "caption_non_audio"
    if has_audio:
        return "audio_non_caption"
    return "motion_non_caption"


TASK_BUCKET_TASKS = {
    bucket: [task for task in ALL_TASKS if task_bucket_name(task) == bucket]
    for bucket in TASK_BUCKETS
}


@DATASETS.register_module(force=True)
class MotionhubMultiTaskMultiAgentDataset(MotionHubSingleAgentDataset):
    collate_fn = staticmethod(flexible_collate)
    SUPPORTED_TASK_MODE = ["auto", "preset"]
    SUPPORTED_TASK_BUCKET_MODE = ["none", "modality"]

    def __init__(
        self,
        motion_key: str = "smplx",
        data_dir: str = "data/motionhub",
        anno_file: str = "data/motionhub/annotations/all/train.json",
        pipeline: Union[Dict, BaseTransform, List[Union[Dict, BaseTransform]]] = None,
        refetch: bool = True,
        verbose: bool = False,
        task_mode: str = "auto",
        task_bucket_mode: str = "none",
        preset_tasks: Optional[List[str]] = None,
        log_task_iter: int = 10000,
        num_person: Optional[int] = None,
        require_caption: bool = False,
        require_caption_file: bool = False,
        resolve_missing_nested_paths: bool = False,
    ):
        self.num_person = num_person
        self.require_caption = require_caption
        self.require_caption_file = require_caption_file
        self.resolve_missing_nested_paths = resolve_missing_nested_paths
        self._resolved_path_cache: Dict[Optional[str], Optional[str]] = {}
        self._direct_path_index_cache: Dict[str, Dict[str, str]] = {}
        self._nested_path_index_cache: Dict[str, Dict[str, str]] = {}
        self._task_sampler_validate_paths = False
        self._task_sampler_max_duration: Optional[float] = None
        self._task_sampler_valid_indices: List[int] = []

        super().__init__(
            motion_key=motion_key,
            data_dir=data_dir,
            anno_file=anno_file,
            pipeline=pipeline,
            refetch=refetch,
            verbose=verbose,
        )
        assert (
            task_mode in self.SUPPORTED_TASK_MODE
        ), f"task_mode must be in {self.SUPPORTED_TASK_MODE}"
        assert (
            task_bucket_mode in self.SUPPORTED_TASK_BUCKET_MODE
        ), f"task_bucket_mode must be in {self.SUPPORTED_TASK_BUCKET_MODE}"

        self.task_mode = task_mode
        self.task_bucket_mode = task_bucket_mode
        if task_mode == "preset":
            self.preset_tasks = abbr_list_to_task_list(preset_tasks)
        self.log_task_iter = log_task_iter

        # ``spawn`` dataloader workers require the dataset to be pickleable.
        # A local lambda breaks pickling, while ``int`` keeps the same zero
        # default behaviour.
        self.task_counter = defaultdict(int)
        # if num_person is not None, then only use num_person agents
        self._task_bucket_mask_cache: List[Optional[int]] = [None] * len(self.data_list)
        self._task_eligible_indices_cache: Dict[Tuple[Tuple[str, ...], bool, Optional[float]], Dict[str, List[int]]] = {}

        self._inject_dataset_into_compose_transforms()

    def _inject_dataset_into_compose_transforms(self):
        """Wire dataset reference into ComposeMultiPerson transforms in the pipeline."""
        from motius.datasets.motion.motionhub.transforms.compose_multi_person import ComposeMultiPerson

        if hasattr(self, "pipeline") and hasattr(self.pipeline, "transforms"):
            for transform in self.pipeline.transforms:
                if isinstance(transform, ComposeMultiPerson):
                    transform.set_dataset(self)

    def _build_direct_path_index(self, parent_dir: str) -> Dict[str, str]:
        cached = self._direct_path_index_cache.get(parent_dir)
        if cached is not None:
            return cached

        index: Dict[str, str] = {}
        if os.path.isdir(parent_dir):
            try:
                for entry in os.scandir(parent_dir):
                    if entry.is_file():
                        index.setdefault(entry.name, entry.path)
            except OSError:
                pass
        self._direct_path_index_cache[parent_dir] = index
        return index

    def _build_nested_path_index(self, parent_dir: str) -> Dict[str, str]:
        cached = self._nested_path_index_cache.get(parent_dir)
        if cached is not None:
            return cached

        index: Dict[str, str] = {}
        if os.path.isdir(parent_dir):
            try:
                for entry in os.scandir(parent_dir):
                    if not entry.is_dir():
                        continue
                    try:
                        for child in os.scandir(entry.path):
                            if child.is_file():
                                index.setdefault(child.name, child.path)
                    except OSError:
                        continue
            except OSError:
                pass
        self._nested_path_index_cache[parent_dir] = index
        return index

    @staticmethod
    def _fallback_rel_paths(rel_path: str) -> List[str]:
        candidates: List[str] = []
        parts = rel_path.split("/")
        for idx, part in enumerate(parts):
            if part != "smplx_55":
                continue
            fallback = list(parts)
            fallback[idx] = "smplh_52"
            candidates.append("/".join(fallback))
        return candidates

    def _resolve_data_path(self, rel_path: Optional[str]) -> Optional[str]:
        if rel_path is None:
            return None
        cached = self._resolved_path_cache.get(rel_path)
        if rel_path in self._resolved_path_cache:
            return cached

        path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_dir, rel_path)
        path = os.path.normpath(path)
        parent_dir = os.path.dirname(path)
        basename = os.path.basename(path)
        resolved = self._build_direct_path_index(parent_dir).get(basename)
        if resolved is None and self.resolve_missing_nested_paths:
            nested_index = self._build_nested_path_index(parent_dir)
            resolved = nested_index.get(basename)

        if resolved is None:
            for fallback_rel_path in self._fallback_rel_paths(rel_path):
                fallback_path = os.path.join(self.data_dir, fallback_rel_path)
                fallback_parent = os.path.dirname(fallback_path)
                fallback_basename = os.path.basename(fallback_path)
                resolved = self._build_direct_path_index(fallback_parent).get(
                    fallback_basename
                )
                if resolved is None and self.resolve_missing_nested_paths:
                    nested_index = self._build_nested_path_index(fallback_parent)
                    resolved = nested_index.get(fallback_basename)
                if resolved is not None:
                    break

        self._resolved_path_cache[rel_path] = resolved
        return resolved

    @staticmethod
    def _parse_index_extra(extra: Tuple) -> Tuple[Optional[str], Optional[str]]:
        if not extra:
            return None, None
        if len(extra) >= 2 and extra[0] == "task":
            return None, extra[1]
        first = extra[0]
        if isinstance(first, str) and first in ABBR_TASK_MAPPING:
            return None, first
        return first, None

    @staticmethod
    def _inline_caption(data_info: Dict) -> Optional[str]:
        caption = data_info.get("caption")
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
        caption_list = data_info.get("caption_list")
        if isinstance(caption_list, (list, tuple)):
            for candidate in caption_list:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return None

    @staticmethod
    def _get_dist_info() -> Tuple[int, int]:
        try:
            from mmengine import dist

            if dist.is_distributed():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass

        try:
            import torch.distributed as torch_dist

            if torch_dist.is_available() and torch_dist.is_initialized():
                return torch_dist.get_rank(), torch_dist.get_world_size()
        except Exception:
            pass

        rank = os.environ.get("RANK")
        world_size = os.environ.get("WORLD_SIZE")
        if rank is not None and world_size is not None:
            try:
                return int(rank), int(world_size)
            except ValueError:
                pass

        local_rank = os.environ.get("LOCAL_RANK")
        node_rank = os.environ.get("NODE_RANK")
        nnodes = os.environ.get("NNODES")
        if local_rank is not None and node_rank is not None and nnodes is not None:
            try:
                local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
                if local_world_size is None:
                    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    local_world_size = (
                        len([item for item in visible_devices.split(",") if item.strip()])
                        if visible_devices
                        else 1
                    )
                local_world_size = int(local_world_size)
                global_rank = int(node_rank) * local_world_size + int(local_rank)
                return global_rank, int(nnodes) * local_world_size
            except ValueError:
                pass

        return 0, 1

    def _task_pool_cache_path(
        self,
        task_abbrs: Tuple[str, ...],
        validate_paths: bool,
        max_duration: Optional[float],
    ) -> str:
        anno_path = os.path.abspath(self.anno_file)
        try:
            anno_mtime = os.path.getmtime(anno_path)
            anno_size = os.path.getsize(anno_path)
        except OSError:
            anno_mtime = None
            anno_size = None

        preset_abbrs = tuple(
            getattr(task, "abbr", task.__name__)
            for task in getattr(self, "preset_tasks", [])
        )
        signature = repr(
            {
                "version": 5,
                "anno_path": anno_path,
                "anno_mtime": anno_mtime,
                "anno_size": anno_size,
                "data_dir": os.path.abspath(self.data_dir),
                "motion_key": self.motion_key,
                "num_person": self.num_person,
                "task_mode": self.task_mode,
                "task_bucket_mode": self.task_bucket_mode,
                "preset_tasks": preset_abbrs,
                "task_abbrs": task_abbrs,
                "validate_paths": validate_paths,
                "max_duration": max_duration,
                "resolve_missing_nested_paths": self.resolve_missing_nested_paths,
                "data_len": len(self.data_list),
            }
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
        cache_dir = os.environ.get(
            "MOTIUS_TASK_POOL_CACHE_DIR",
            os.path.join(os.getcwd(), "work_dirs", "task_pool_cache"),
        )
        return os.path.join(cache_dir, f"{digest}.pkl")

    def _load_task_pool_cache(self, cache_path: str) -> Optional[Dict[str, List[int]]]:
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        pools = payload.get("pools")
        if not isinstance(pools, dict):
            return None
        return pools

    def _save_task_pool_cache(
        self, cache_path: str, pools: Dict[str, List[int]]
    ) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump({"pools": pools}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)

    def load_data_list(self) -> List[dict]:
        """Copied from mmengine.dataset.based_dataset.BaseDataset
        Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`meta_info`
        and ``meta_info`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = mmengine.load(self.anno_file)
        if not isinstance(annotations, dict):
            raise TypeError(
                f"The annotations loaded from annotation file "
                f"should be a dict, but got {type(annotations)}!"
            )
        meta_info = annotations.get("meta_info", annotations.get("meta"))
        if "data_list" not in annotations or meta_info is None:
            raise ValueError("Annotation must have data_list and meta_info/meta keys")

        for k, v in meta_info.items():
            self._metainfo.setdefault(k, v)

        raw_data_list = annotations["data_list"]

        is_main = True
        try:
            from mmengine import dist

            is_main = (not dist.is_distributed()) or (dist.get_rank() == 0)
        except ImportError:
            is_main = True

        iterator = (
            tqdm(raw_data_list.values(), desc="Loading data_list")
            if is_main
            else raw_data_list.values()
        )

        data_list = []
        skipped_no_caption = 0
        skipped_missing_caption_file = 0
        for data_info in iterator:
            assert isinstance(data_info, dict)
            caption_path = data_info.get(
                "hierarchical_caption_path", data_info.get("caption_path")
            )
            inline_caption = self._inline_caption(data_info)
            if self.require_caption:
                if not caption_path and inline_caption is None:
                    skipped_no_caption += 1
                    continue
                if self.require_caption_file and caption_path and inline_caption is None:
                    resolved_caption_path = os.path.join(self.data_dir, caption_path)
                    if not os.path.exists(resolved_caption_path):
                        skipped_missing_caption_file += 1
                        continue

            # skip multi-person data
            motion_path: Union[str, List[str]] = data_info[f"{self.motion_key}_path"]

            if isinstance(motion_path, str):
                if self.num_person is not None and self.num_person != 1:
                    continue

            if self.num_person is not None:
                if (
                    isinstance(motion_path, list)
                    and len(motion_path) != self.num_person
                ):
                    continue
            data_list.append(data_info)

        skip_msg = ""
        if self.require_caption:
            skip_msg = (
                f" (skipped {skipped_no_caption} without caption"
                f", {skipped_missing_caption_file} missing caption files)"
            )
        print_log(f"Loaded {len(data_list)} samples from {self.anno_file}{skip_msg}")
        return data_list

    @override
    def prepare_data(self, idx: int) -> dict:
        raw_idx, extra = self._split_index(idx)
        task_bucket, fixed_task_abbr = self._parse_index_extra(extra)

        raw_data_info = self.data_list[raw_idx]

        # load motion
        raw_motion_path = raw_data_info[f"{self.motion_key}_path"]

        # load caption
        caption_path = raw_data_info.get(
            "hierarchical_caption_path", raw_data_info.get("caption_path")
        )
        caption_path = self._resolve_data_path(caption_path)

        # multi agent or single agent
        if isinstance(raw_motion_path, str):
            motion_path = self._resolve_data_path(raw_motion_path)
            num_person = 1
        else:
            assert isinstance(raw_motion_path, list), "motion_path must be a list"
            resolved_motion_paths = [
                self._resolve_data_path(path) for path in raw_motion_path
            ]
            motion_path = (
                resolved_motion_paths
                if all(path is not None for path in resolved_motion_paths)
                else None
            )
            num_person = len(raw_motion_path)

        person_caption_paths = raw_data_info.get(
            "person_caption_paths", raw_data_info.get("sep_hierarchical_caption_path")
        )
        if isinstance(person_caption_paths, str):
            person_caption_paths = [person_caption_paths]
        if isinstance(person_caption_paths, (list, tuple)):
            resolved_caption_paths = []
            for path in person_caption_paths:
                if not path:
                    continue
                path = os.path.join(self.data_dir, path)
                if os.path.exists(path):
                    resolved_caption_paths.append(path)
            person_caption_paths = (
                resolved_caption_paths
                if len(resolved_caption_paths) == num_person
                else None
            )
        else:
            person_caption_paths = None

        # load music
        music_path = raw_data_info.get("music_path", None)
        music_path = self._resolve_data_path(music_path)
        # Load speech related
        audio_path = raw_data_info.get("audio_path", None)
        audio_path = self._resolve_data_path(audio_path)

        speech_script_path = raw_data_info.get("speech_script_path", None)
        speech_script_path = self._resolve_data_path(speech_script_path)

        # source_motion_path: for editing pairs (e.g. PerMo Neutral→Emotion)
        source_motion_path = raw_data_info.get("source_motion_path", None)
        source_motion_path = self._resolve_data_path(source_motion_path)

        data_info = {
            "sample_pathway": (
                "task_uniform" if fixed_task_abbr else "data_proportional"
            ),
            "source_audit_group": raw_data_info.get("source_audit_group"),
            "num_person": num_person,
            "motion_path": motion_path,
            "subset": raw_data_info["subset"],
            "data_src": raw_data_info.get("data_src", raw_data_info.get("source", raw_data_info["subset"])),
            "source": raw_data_info.get("source", raw_data_info.get("data_src", raw_data_info["subset"])),
            "fps": raw_data_info["fps"],
            "has_hand": raw_data_info["has_hand"],
            "duration": raw_data_info["duration"],
            "num_frames": raw_data_info["num_frames"],
            # text related
            "caption_path": caption_path,
            "caption": self._inline_caption(raw_data_info),
            "caption_list": raw_data_info.get("caption_list"),
            "person_caption_paths": person_caption_paths,
            # sound related
            "sr": raw_data_info.get("sr"),
            # music related
            "music_path": music_path,
            "genre": raw_data_info.get("genre"),
            # language related
            "language": raw_data_info.get("language"),
            "audio_path": audio_path,
            "speech_script_path": speech_script_path,
            "speaker_id": raw_data_info.get("speaker_id"),
            # editing pair: source motion for Neutral→Emotion style transfer
            "source_motion_path": source_motion_path,
            # deterministic task tag for overfit/debug datasets
            "overfit_task": raw_data_info.get("overfit_task"),
            "overfit_source_key": raw_data_info.get("overfit_source_key"),
            "overfit_source_annotation": raw_data_info.get("overfit_source_annotation"),
            "overfit_multi_kind": raw_data_info.get("overfit_multi_kind"),
            "_motion_audio_crop_start": raw_data_info.get("_motion_audio_crop_start"),
            "_motion_audio_crop_duration": raw_data_info.get("_motion_audio_crop_duration"),
            "_motion_audio_crop_start_frame": raw_data_info.get(
                "_motion_audio_crop_start_frame"
            ),
            "_motion_audio_crop_num_frames": raw_data_info.get(
                "_motion_audio_crop_num_frames"
            ),
        }
        candidate_tasks = self._resolve_candidate_tasks(task_bucket)
        if fixed_task_abbr:
            if fixed_task_abbr not in ABBR_TASK_MAPPING:
                raise KeyError(
                    f"Unknown fixed task={fixed_task_abbr!r}; "
                    f"available tasks={sorted(ABBR_TASK_MAPPING)}"
                )
            enabled_tasks = {
                task.abbr: task for task in candidate_tasks
            }
            if fixed_task_abbr not in enabled_tasks:
                raise ValueError(
                    f"fixed task={fixed_task_abbr!r} is not enabled by "
                    f"task_mode/task_bucket_mode for data {motion_path}"
                )
            candidate_tasks = [enabled_tasks[fixed_task_abbr]]
        overfit_task_abbr = raw_data_info.get("overfit_task")
        if overfit_task_abbr:
            if overfit_task_abbr not in ABBR_TASK_MAPPING:
                raise KeyError(
                    f"Unknown overfit_task={overfit_task_abbr!r}; "
                    f"available tasks={sorted(ABBR_TASK_MAPPING)}"
                )
            enabled_tasks = {
                task.abbr: task for task in candidate_tasks
            }
            if overfit_task_abbr not in enabled_tasks:
                raise ValueError(
                    f"overfit_task={overfit_task_abbr!r} is not enabled by "
                    f"task_mode/task_bucket_mode for data {motion_path}"
                )
            candidate_tasks = [enabled_tasks[overfit_task_abbr]]
        # determine what tasks can be trained on this data
        available_tasks = self.assign_task_for_data(data_info, candidate_tasks)
        # randomly assign a task for the data info
        if available_tasks:
            task = (
                available_tasks[0]
                if overfit_task_abbr or fixed_task_abbr
                else random.choice(available_tasks)
            )

            data_info["task"] = task
            self.task_counter[task.abbr] += 1

            if sum(self.task_counter.values()) % self.log_task_iter == 0:
                print_log(self.task_counter, logger="current")
            return data_info
        raise ValueError(
            f"No available task in {[task.abbr for task in candidate_tasks]} "
            f"for data {motion_path}; availability="
            f"{{'motion': {motion_path is not None}, "
            f"'caption_path': {caption_path is not None}, "
            f"'inline_caption': {data_info['caption'] is not None}, "
            f"'num_person': {data_info['num_person'] is not None}}}"
        )

    def _resolve_candidate_tasks(
        self, task_bucket: Optional[str] = None
    ) -> List[Type[BaseTask]]:
        if self.task_mode == "preset":
            candidate_tasks = list(self.preset_tasks)
        else:
            candidate_tasks = list(ALL_TASKS)

        if (
            task_bucket is not None
            and self.task_bucket_mode != "none"
            and task_bucket in TASK_BUCKET_TASKS
        ):
            bucket_task_abbrs = {
                task.abbr for task in TASK_BUCKET_TASKS[task_bucket]
            }
            candidate_tasks = [
                task for task in candidate_tasks
                if task.abbr in bucket_task_abbrs
            ]

        return candidate_tasks

    def _build_bucket_probe_data_info(
        self,
        raw_data_info: Dict,
        validate_paths: bool = False,
        required_modal_names: Optional[set] = None,
    ) -> Dict:
        motion_path = raw_data_info[f"{self.motion_key}_path"]
        num_person = 1 if isinstance(motion_path, str) else len(motion_path)
        caption_path = raw_data_info.get(
            "hierarchical_caption_path", raw_data_info.get("caption_path")
        )
        music_path = raw_data_info.get("music_path")
        audio_path = raw_data_info.get("audio_path")
        speech_script_path = raw_data_info.get("speech_script_path")
        if validate_paths:
            if isinstance(motion_path, str):
                motion_path = self._resolve_data_path(motion_path)
            else:
                resolved_motion_paths = [
                    self._resolve_data_path(path) for path in motion_path
                ]
                motion_path = (
                    resolved_motion_paths
                    if all(path is not None for path in resolved_motion_paths)
                    else None
                )
            required_modal_names = required_modal_names or {
                "caption",
                "music",
                "past_music",
                "future_music",
                "audio",
                "speech_script",
            }
            if "caption" in required_modal_names:
                caption_path = self._resolve_data_path(caption_path)
            if required_modal_names & {"music", "past_music", "future_music"}:
                music_path = self._resolve_data_path(music_path)
            if "audio" in required_modal_names:
                audio_path = self._resolve_data_path(audio_path)
            if "speech_script" in required_modal_names:
                speech_script_path = self._resolve_data_path(speech_script_path)
        return {
            "num_person": num_person,
            "motion_path": motion_path,
            "duration": raw_data_info.get("duration"),
            "num_frames": raw_data_info.get("num_frames"),
            "caption_path": caption_path,
            "caption": self._inline_caption(raw_data_info),
            "caption_list": raw_data_info.get("caption_list"),
            "music_path": music_path,
            "genre": raw_data_info.get("genre"),
            "audio_path": audio_path,
            "speech_script_path": speech_script_path,
            "sr": raw_data_info.get("sr"),
        }

    @staticmethod
    def _row_allows_fixed_task(raw_data_info: Dict, task_abbr: str) -> bool:
        """Keep deterministic debug rows in their declared task pool."""
        overfit_task = raw_data_info.get("overfit_task")
        return overfit_task is None or overfit_task == task_abbr

    def _infer_task_bucket_mask(self, raw_idx: int) -> int:
        if self.task_bucket_mode == "none":
            return 0

        raw_data_info = self.data_list[raw_idx]
        probe_data_info = self._build_bucket_probe_data_info(raw_data_info)
        mask = 0
        for bucket_idx, bucket_name in enumerate(TASK_BUCKETS):
            available = self.assign_task_for_data(
                probe_data_info,
                self._resolve_candidate_tasks(bucket_name),
            )
            if available:
                mask |= 1 << bucket_idx
        return mask

    def get_task_bucket_names(self, raw_idx: int) -> Tuple[str, ...]:
        if self.task_bucket_mode == "none":
            return tuple()

        mask = self._task_bucket_mask_cache[raw_idx]
        if mask is None:
            mask = self._infer_task_bucket_mask(raw_idx)
            self._task_bucket_mask_cache[raw_idx] = mask

        return tuple(
            bucket_name
            for bucket_idx, bucket_name in enumerate(TASK_BUCKETS)
            if mask & (1 << bucket_idx)
        )

    def sample_refetch_index(self, *extra) -> int:
        task_bucket, fixed_task_abbr = self._parse_index_extra(extra)
        if fixed_task_abbr:
            pool = []
            found_cached_pool = False
            for (task_abbrs, validate_paths, max_duration), pools in (
                self._task_eligible_indices_cache.items()
            ):
                if (
                    fixed_task_abbr in task_abbrs
                    and validate_paths == self._task_sampler_validate_paths
                    and max_duration == self._task_sampler_max_duration
                ):
                    found_cached_pool = True
                    pool = pools.get(fixed_task_abbr, [])
                    break
            if not pool and not found_cached_pool:
                pools = self.get_task_eligible_indices(
                    [fixed_task_abbr],
                    validate_paths=self._task_sampler_validate_paths,
                    max_duration=self._task_sampler_max_duration,
                )
                pool = pools.get(fixed_task_abbr, [])
            if pool:
                return random.choice(pool)

        if self._task_sampler_valid_indices:
            return random.choice(self._task_sampler_valid_indices)

        if not task_bucket or self.task_bucket_mode == "none":
            return random.randint(0, len(self.data_list) - 1)

        for _ in range(32):
            idx = random.randint(0, len(self.data_list) - 1)
            if task_bucket in self.get_task_bucket_names(idx):
                return idx

        return random.randint(0, len(self.data_list) - 1)

    def get_task_eligible_indices(
        self,
        task_abbrs: Optional[List[str]] = None,
        validate_paths: bool = True,
        max_duration: Optional[float] = None,
    ) -> Dict[str, List[int]]:
        if task_abbrs is None:
            task_abbrs = sorted(ABBR_TASK_MAPPING)
        task_abbrs = tuple(task_abbrs)
        cache_key = (task_abbrs, validate_paths, max_duration)
        cached = self._task_eligible_indices_cache.get(cache_key)
        if cached is not None:
            return cached

        rank, world_size = self._get_dist_info()
        cache_path = self._task_pool_cache_path(
            task_abbrs, validate_paths, max_duration
        )
        disk_cached = self._load_task_pool_cache(cache_path)
        if disk_cached is not None:
            self._task_eligible_indices_cache[cache_key] = disk_cached
            if rank == 0:
                print_log(
                    f"Loaded task eligible index cache from {cache_path}",
                    logger="current",
                )
            return disk_cached

        if world_size > 1 and rank != 0:
            deadline = time.time() + float(
                os.environ.get("MOTIUS_TASK_POOL_CACHE_WAIT_SEC", "1800")
            )
            while time.time() < deadline:
                disk_cached = self._load_task_pool_cache(cache_path)
                if disk_cached is not None:
                    self._task_eligible_indices_cache[cache_key] = disk_cached
                    return disk_cached
                time.sleep(2.0)
            print_log(
                f"Timed out waiting for task eligible index cache at {cache_path}; "
                "building locally.",
                logger="current",
                level="WARNING",
            )

        task_classes = {abbr: ABBR_TASK_MAPPING[abbr] for abbr in task_abbrs}
        pools: Dict[str, List[int]] = {abbr: [] for abbr in task_abbrs}
        task_required_modals = {
            abbr: {modal.name for modal in task.essential_modality()}
            for abbr, task in task_classes.items()
        }
        required_modal_names = set()
        for modal_names in task_required_modals.values():
            required_modal_names.update(modal_names)
        started_at = time.time()
        if rank == 0:
            print_log(
                "Building task eligible index cache: "
                f"tasks={list(task_abbrs)}, validate_paths={validate_paths}, "
                f"max_duration={max_duration}, data_len={len(self.data_list)}, "
                f"cache_path={cache_path}",
                logger="current",
            )
        for raw_idx, raw_data_info in enumerate(self.data_list):
            if max_duration is not None:
                duration = raw_data_info.get("duration")
                if duration is None or float(duration) > max_duration:
                    continue
            probe_data_info = self._build_bucket_probe_data_info(
                raw_data_info,
                validate_paths=validate_paths,
                required_modal_names=required_modal_names,
            )
            for abbr, task in task_classes.items():
                if not self._row_allows_fixed_task(raw_data_info, abbr):
                    continue
                if self.assign_task_for_data(probe_data_info, [task]):
                    pools[abbr].append(raw_idx)

        try:
            self._save_task_pool_cache(cache_path, pools)
            if rank == 0:
                print_log(
                    "Saved task eligible index cache to "
                    f"{cache_path} in {time.time() - started_at:.1f}s; "
                    f"sizes={{{', '.join(f'{abbr}: {len(indices)}' for abbr, indices in pools.items())}}}",
                    logger="current",
                )
        except Exception as exc:
            if rank == 0:
                print_log(
                    f"Failed to save task eligible index cache to {cache_path}: {exc}",
                    logger="current",
                    level="WARNING",
                )

        self._task_eligible_indices_cache[cache_key] = pools
        return pools

    def assign_task_for_data(
        self, data_info: Dict, candidate_tasks: Optional[List[Type[Modality]]] = None
    ) -> List[BaseTask]:
        """Assign the task for the data info

        :param data_info: original data info
        :param candidate_tasks: preset tasks, if None, use all tasks
        :return: The task-specific data info.
        """
        tasks = []
        candidate_tasks: List[BaseTask] = (
            ALL_TASKS if candidate_tasks is None else candidate_tasks
        )

        for task in candidate_tasks:
            essential_modals: List[Type[Modality]] = task.essential_modality()
            # check if all required modalities exist
            for modal in essential_modals:
                modal_exist = False
                candidate_keys = modal.load_keys

                for key in candidate_keys:
                    direct_value = data_info.get(key)
                    path_value = data_info.get(f"{key}_path")
                    if direct_value is not None or path_value is not None:
                        modal_exist = True
                        break
                # if any modality is not found, skip this task
                if not modal_exist:
                    break

            if modal_exist:
                tasks.append(task)

        return tasks
