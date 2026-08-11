"""Small dataset union wrapper for mixed HYMotion training curricula."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from motius.datasets.motion.motionhub.flexible_collate import flexible_collate
from motius.registry import DATASETS


@DATASETS.register_module(force=True)
class MotionDatasetUnion:
    """Concatenate heterogeneous datasets while exposing a sampler-friendly data_list.

    The runner's weighted sampler reads ``dataset.data_list`` and matches group
    substrings against each entry's ``subset``.  PyTorch's generic ConcatDataset
    hides that metadata, so this wrapper keeps a flattened metadata list and
    optionally prefixes each child subset, while delegating sample preparation to
    the original child dataset.
    """

    collate_fn = staticmethod(flexible_collate)

    def __init__(
        self,
        datasets: Sequence[Dict[str, Any]],
        subset_prefixes: Optional[Sequence[str]] = None,
    ):
        if not datasets:
            raise ValueError("MotionDatasetUnion requires at least one child dataset")
        if subset_prefixes is not None and len(subset_prefixes) != len(datasets):
            raise ValueError(
                f"subset_prefixes length {len(subset_prefixes)} must match "
                f"datasets length {len(datasets)}"
            )

        self.datasets = [DATASETS.build(deepcopy(cfg)) for cfg in datasets]
        self.subset_prefixes = list(subset_prefixes or [None] * len(self.datasets))
        self.index_map: List[Tuple[int, int]] = []
        self.data_list: List[Dict[str, Any]] = []

        for ds_idx, dataset in enumerate(self.datasets):
            prefix = self.subset_prefixes[ds_idx]
            child_data_list = getattr(dataset, "data_list", None)
            for local_idx in range(len(dataset)):
                self.index_map.append((ds_idx, local_idx))
                if child_data_list is not None and local_idx < len(child_data_list):
                    item = deepcopy(child_data_list[local_idx])
                else:
                    item = {}
                subset = str(item.get("subset", f"dataset_{ds_idx}"))
                if prefix:
                    item["subset"] = f"{prefix}:{subset}"
                    item["union_subset_prefix"] = prefix
                item["_union_dataset_idx"] = ds_idx
                item["_union_local_idx"] = local_idx
                self.data_list.append(item)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx = self.index_map[idx]
        return self.datasets[ds_idx][local_idx]
