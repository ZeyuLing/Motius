# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Sequence

from mmcv.transforms.base import BaseTransform
import numpy as np
import torch

from motius.datasets.motion.motionhub.common import convert_to_tensor
from motius.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class PackInputs(BaseTransform):
    """Pack selected fields into the flat batch dict expected by HFTrainer."""

    def __init__(
        self,
        keys: Tuple[List[str], str] = ["motion"],
        meta_keys: Tuple[List[str], str] = [],
        data_keys: Tuple[List[str], str] = [],
        # For multi-task mixied training, if a key in `keys` is not found in
        # `results`, set the value to `None`. So that the keys in a batch will be
        # aligned.
        set_dummy_value: bool = False,
        dummy_value: object = None,
    ) -> None:

        assert keys is not None, "keys in PackInputs can not be None."
        assert data_keys is not None, "data_keys in PackInputs can not be None."
        assert meta_keys is not None, "meta_keys in PackInputs can not be None."

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys, List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys, List) else [meta_keys]
        self.meta_keys = self.meta_keys

        self.set_dummy_value = set_dummy_value
        self.dummy_value = dummy_value

    def transform(self, results: dict) -> dict:
        packed = {}
        for k in self.keys:
            value = results.get(k, None)
            if value is not None:
                if isinstance(value, np.ndarray):
                    packed[k] = torch.from_numpy(value)
                else:
                    packed[k] = value
            else:
                if self.set_dummy_value:
                    packed[k] = self.dummy_value

        for k in self.meta_keys + self.data_keys:
            if k in results:
                value = results[k]
                if isinstance(value, np.ndarray):
                    packed[k] = torch.from_numpy(value)
                else:
                    packed[k] = value
        return packed

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str


@TRANSFORMS.register_module(force=True)
class ToTensor(BaseTransform):
    """Convert some results to :obj:`torch.Tensor` by given keys.
    If the key is a dict, transform its values to :obj:`torch.Tensor`

    Required keys:

    - all these keys in `keys`

    Modified Keys:

    - all these keys in `keys`

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = keys

    def transform(self, results: dict) -> dict:
        """Transform function to convert data to `torch.Tensor`.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: `keys` in results will be updated.
        """
        for key in self.keys:

            key_list = key.split(".")
            cur_item = results
            for i in range(len(key_list)):
                if key_list[i] not in cur_item:
                    raise KeyError(f"Can not find key {key}")
                if i == len(key_list) - 1:
                    cur_item[key_list[i]] = convert_to_tensor(cur_item[key_list[i]])
                    break
                cur_item = cur_item[key_list[i]]

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(keys={self.keys})"
