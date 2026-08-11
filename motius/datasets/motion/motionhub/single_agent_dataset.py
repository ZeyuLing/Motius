import logging
import os
import random
from typing import Any, Dict, List, Union

import mmengine
from tqdm import tqdm
from mmengine.dataset import BaseDataset
from mmengine import print_log

from motius.datasets.motion.motionhub.flexible_collate import flexible_collate
from motius.registry import DATASETS


@DATASETS.register_module(force=True)
class MotionHubSingleAgentDataset(BaseDataset):
    collate_fn = staticmethod(flexible_collate)

    @staticmethod
    def _split_index(idx):
        if isinstance(idx, tuple) and len(idx) >= 1:
            return idx[0], idx[1:]
        return idx, ()

    @staticmethod
    def _merge_index(raw_idx: int, extra) -> Union[int, tuple]:
        if not extra:
            return raw_idx
        return (raw_idx, *extra)

    def __init__(
        self,
        motion_key: str = "smplx",
        data_dir: str = "data/motionhub",
        anno_file: str = "data/motionhub/annotations/all/train.json",
        pipeline: Union[Dict, Any, List[Union[Dict, Any]]] = None,
        refetch: bool = False,
        max_refetch: int = 100,
        verbose: bool = True,
    ):
        self.motion_key = motion_key
        self.data_dir = data_dir
        self.anno_file = anno_file
        self.refetch = refetch
        self.max_refetch = max_refetch  # 单次 __getitem__ 内最多 refetch 次数，超过则抛出异常终止
        self.verbose = verbose
        super().__init__(
            ann_file=anno_file,
            metainfo=None,
            data_root=data_dir,
            data_prefix={},
            serialize_data=False,
            pipeline=list(pipeline) if pipeline is not None else [],
            test_mode=False,
            lazy_init=False,
            max_refetch=max_refetch,
        )

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
        for data_info in iterator:
            assert isinstance(data_info, dict)
            # skip multi-person data
            if not isinstance(data_info[f"{self.motion_key}_path"], str):
                continue
            data_list.append(data_info)

        print_log(f"Loaded {len(data_list)} samples from {self.anno_file}")

        return data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int, _refetch_depth: int = 0) -> dict:
        """The returned Dict will be like:
        {
            "motion": torch.rand([T, C]),
            "motion_metadata": {
                "motion_path": motion_path,
                "duration": T / fps,
                "num_frames": T,
                "fps": fps,
            },
        }
        _refetch_depth: 内部使用，表示当前 refetch 深度，超过 max_refetch 则不再重试并抛错。
        """
        try:
            sample = self.prepare_data(idx)
            sample = self.pipeline(sample)

            # If pipeline returns None, treat it as invalid sample and trigger refetch
            if sample is None:
                if not self.refetch:
                    raise ValueError(f"Pipeline returned None for idx={idx} and refetch is disabled")
                if _refetch_depth >= self.max_refetch:
                    print_log(
                        f"Refetch exceeded max_refetch={self.max_refetch} (failed idx={idx}, pipeline returned None), raising.",
                        level=logging.ERROR,
                    )
                    raise ValueError(f"Pipeline returned None for idx={idx} after {self.max_refetch} refetch attempts")
                if self.verbose:
                    print_log(
                        f"Pipeline returned None for idx={idx} (refetch {_refetch_depth + 1}/{self.max_refetch}): "
                        f"fetching another instead",
                        level=logging.WARNING,
                    )
                raw_idx, extra = self._split_index(idx)
                if extra and hasattr(self, "sample_refetch_index"):
                    new_raw_idx = self.sample_refetch_index(*extra)
                else:
                    new_raw_idx = random.randint(0, len(self.data_list) - 1)
                    if new_raw_idx == raw_idx:
                        new_raw_idx = (raw_idx + 1) % len(self.data_list)
                new_idx = self._merge_index(new_raw_idx, extra)
                return self.__getitem__(new_idx, _refetch_depth + 1)

            return sample

        except Exception as e:
            if not self.refetch:
                raise e
            if _refetch_depth >= self.max_refetch:
                print_log(
                    f"Refetch exceeded max_refetch={self.max_refetch} (failed idx={idx}), raising.",
                    level=logging.ERROR,
                )
                raise e
            if self.verbose:
                print_log(
                    f"Get error when loading idx={idx} (refetch {_refetch_depth + 1}/{self.max_refetch}): "
                    f"fetching another instead, error: {e}",
                    level=logging.WARNING,
                )
            raw_idx, extra = self._split_index(idx)
            if extra and hasattr(self, "sample_refetch_index"):
                new_raw_idx = self.sample_refetch_index(*extra)
            else:
                new_raw_idx = random.randint(0, len(self.data_list) - 1)
                if new_raw_idx == raw_idx:
                    new_raw_idx = (raw_idx + 1) % len(self.data_list)
            new_idx = self._merge_index(new_raw_idx, extra)
            return self.__getitem__(new_idx, _refetch_depth + 1)

    def full_init(self):
        super().full_init()

    def prepare_data(self, idx: int) -> dict:
        raw_data_info = self.data_list[idx]
        data_info = {
            "motion_path": os.path.join(
                self.data_dir, raw_data_info[f"{self.motion_key}_path"]
            ),  # cannot be None
        }
        return data_info
