from typing import List
from ..modality import Motion
from .base_task import BaseTask


class MotionPretrainTask(BaseTask):
    abbr = "pretrain"
    description = "Only motion"
    templates = [""]
    optional_input_modality = []
    input_modality: List = []
    output_modality: List = [Motion]
