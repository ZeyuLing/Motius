import os
from typing import Any, Dict, List, Union

from motius.datasets.motion.motionhub.single_agent_dataset import MotionHubSingleAgentDataset
from motius.registry import DATASETS


@DATASETS.register_module()
class MotionHubSingleAgentTextDataset(MotionHubSingleAgentDataset):

    def __init__(
        self,
        motion_key: str = "smplx",
        caption_key: str = "hierarchical_caption",
        data_dir: str = "data/motionhub",
        anno_file: str = "data/motionhub/annotations/all/train.json",
        pipeline: Union[Dict, Any, List[Union[Dict, Any]]] = None,
        refetch=False,
        verbose: bool = True,
    ):
        super().__init__(
            motion_key=motion_key,
            data_dir=data_dir,
            anno_file=anno_file,
            pipeline=pipeline,
            refetch=refetch,
            verbose=verbose,
        )
        self.caption_key = caption_key

    def prepare_data(self, idx: int) -> dict:
        raw_data_info = self.data_list[idx]
        data_info = {
            "motion_path": os.path.join(
                self.data_dir, raw_data_info[f"{self.motion_key}_path"]
            ),  # cannot be None
            "caption_path": os.path.join(
                self.data_dir, raw_data_info[f"{self.caption_key}_path"]
            ),  # cannot be None
        }
        return data_info
