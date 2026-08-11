import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Union

import mmengine
from mmengine import print_log
from tqdm import tqdm

from motius.datasets.motion.motionhub.flexible_collate import flexible_collate
from motius.datasets.motion.motionhub.single_agent_dataset import MotionHubSingleAgentDataset
from motius.registry import DATASETS


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


@DATASETS.register_module(force=True)
class MotionHubReconstructionDataset(MotionHubSingleAgentDataset):
    """Motion-only dataset for tokenizer/VAE reconstruction training.

    Unlike ``MotionhubMultiTaskMultiAgentDataset``, this dataset does not assign
    VerMo tasks.  It only resolves SMPL-H/SMPL-X motion paths, keeps useful
    metadata for logging, and then lets the configured transform pipeline load
    the motion tensor.
    """

    collate_fn = staticmethod(flexible_collate)

    def __init__(
        self,
        motion_key: str = "smplx",
        data_dir: str = "data/motionhub",
        anno_file: Union[str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = "data/motionhub/annotations/all/train.json",
        pipeline: Union[Dict, Any, List[Union[Dict, Any]]] = None,
        refetch: bool = True,
        max_refetch: int = 100,
        verbose: bool = False,
        include_num_persons: Optional[Sequence[int]] = None,
        sample_domain: Optional[str] = None,
    ):
        self._annotation_specs_input = anno_file
        self.include_num_persons = (
            set(int(x) for x in include_num_persons)
            if include_num_persons is not None
            else None
        )
        self.default_sample_domain = sample_domain
        super().__init__(
            motion_key=motion_key,
            data_dir=data_dir,
            anno_file="",
            pipeline=pipeline,
            refetch=refetch,
            max_refetch=max_refetch,
            verbose=verbose,
        )
        self._inject_dataset_into_compose_transforms()

    def _inject_dataset_into_compose_transforms(self) -> None:
        """Wire dataset reference into ComposeMultiPerson transforms."""
        from motius.datasets.motion.motionhub.transforms.compose_multi_person import ComposeMultiPerson

        if hasattr(self, "pipeline") and hasattr(self.pipeline, "transforms"):
            for transform in self.pipeline.transforms:
                if isinstance(transform, ComposeMultiPerson):
                    transform.set_dataset(self)

    @staticmethod
    def _num_person(motion_path) -> int:
        return 1 if isinstance(motion_path, str) else len(motion_path)

    def _iter_annotation_specs(self) -> List[Dict[str, Any]]:
        specs = _as_list(self._annotation_specs_input)
        if not specs:
            raise ValueError("anno_file must not be empty")
        out = []
        for spec in specs:
            if isinstance(spec, str):
                out.append({"path": spec})
            elif isinstance(spec, dict):
                if "path" not in spec:
                    raise KeyError(f"Annotation spec requires a 'path': {spec}")
                out.append(dict(spec))
            else:
                raise TypeError(f"Unsupported annotation spec: {type(spec)!r}")
        return out

    def _include_person(self, n_person: int, spec: Dict[str, Any]) -> bool:
        include = spec.get("include_num_persons", self.include_num_persons)
        if include is None:
            return True
        return n_person in set(int(x) for x in include)

    def _resolve_path(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            resolved = [self._resolve_path(v) for v in value]
            return resolved if all(v is not None for v in resolved) else None
        if os.path.isabs(str(value)):
            return str(value)
        path = os.path.normpath(os.path.join(self.data_dir, str(value)))
        if os.path.exists(path):
            return path

        # Common migration fallback: annotations may still say smplx_55 while
        # the processed files now live under smplh_52.
        parts = str(value).split("/")
        for idx, part in enumerate(parts):
            if part != "smplx_55":
                continue
            fallback = list(parts)
            fallback[idx] = "smplh_52"
            fallback_path = os.path.normpath(os.path.join(self.data_dir, "/".join(fallback)))
            if os.path.exists(fallback_path):
                return fallback_path
        return path

    def load_data_list(self) -> List[dict]:
        data_list: List[dict] = []
        is_main = True
        try:
            from mmengine import dist

            is_main = (not dist.is_distributed()) or (dist.get_rank() == 0)
        except Exception:
            pass

        for spec in self._iter_annotation_specs():
            anno_path = spec["path"]
            annotations = mmengine.load(anno_path)
            if not isinstance(annotations, dict):
                raise TypeError(f"Annotation {anno_path} should be a dict")
            meta_info = annotations.get("meta_info", annotations.get("meta", {}))
            if "data_list" not in annotations:
                raise ValueError(f"Annotation {anno_path} must have data_list")
            for k, v in meta_info.items():
                self._metainfo.setdefault(k, v)

            raw_data_list = annotations["data_list"]
            iterator = (
                tqdm(raw_data_list.items(), desc=f"Loading {anno_path}")
                if is_main
                else raw_data_list.items()
            )
            loaded = 0
            for key, data_info in iterator:
                if not isinstance(data_info, dict):
                    continue
                motion_path = data_info.get(f"{self.motion_key}_path")
                if motion_path is None:
                    continue
                n_person = self._num_person(motion_path)
                if not self._include_person(n_person, spec):
                    continue
                row = dict(data_info)
                row["_annotation_file"] = anno_path
                row["_annotation_key"] = key
                row["_sample_domain"] = spec.get(
                    "sample_domain",
                    self.default_sample_domain or ("real_multi" if n_person > 1 else "single"),
                )
                data_list.append(row)
                loaded += 1
            print_log(f"Loaded {loaded} reconstruction samples from {anno_path}")

        print_log(f"Loaded {len(data_list)} reconstruction samples total")
        return data_list

    def prepare_data(self, idx: int) -> dict:
        raw_data_info = self.data_list[idx]
        raw_motion_path = raw_data_info[f"{self.motion_key}_path"]
        motion_path = self._resolve_path(raw_motion_path)
        n_person = self._num_person(raw_motion_path)

        caption_path = raw_data_info.get(
            "hierarchical_caption_path", raw_data_info.get("caption_path")
        )
        person_caption_paths = raw_data_info.get(
            "person_caption_paths", raw_data_info.get("sep_hierarchical_caption_path")
        )
        if isinstance(person_caption_paths, str):
            person_caption_paths = [person_caption_paths]
        if isinstance(person_caption_paths, (list, tuple)):
            resolved_person_caption_paths = []
            for path in person_caption_paths:
                resolved = self._resolve_path(path)
                if resolved is not None and os.path.exists(resolved):
                    resolved_person_caption_paths.append(resolved)
            person_caption_paths = (
                resolved_person_caption_paths
                if len(resolved_person_caption_paths) == n_person
                else None
            )
        else:
            person_caption_paths = None

        return {
            "num_person": n_person,
            "motion_path": motion_path,
            "subset": raw_data_info.get("subset", raw_data_info.get("source", "unknown")),
            "data_src": raw_data_info.get("data_src", raw_data_info.get("source", raw_data_info.get("subset", "unknown"))),
            "source": raw_data_info.get("source", raw_data_info.get("data_src", raw_data_info.get("subset", "unknown"))),
            "fps": raw_data_info.get("fps", 30),
            "has_hand": raw_data_info.get("has_hand", True),
            "duration": raw_data_info.get("duration"),
            "num_frames": raw_data_info.get("num_frames"),
            "caption_path": self._resolve_path(caption_path),
            "person_caption_paths": person_caption_paths,
            "sample_domain": raw_data_info.get("_sample_domain", "unknown"),
            "annotation_key": raw_data_info.get("_annotation_key"),
            "annotation_file": raw_data_info.get("_annotation_file"),
        }
