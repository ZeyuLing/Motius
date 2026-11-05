from typing import List
from motius.motion.conditioning.vermo.modality import Motion
from motius.motion.conditioning.vermo.task_lib.base_task import BaseTask


class MotionPretrainTask(BaseTask):
    abbr = "pretrain"
    description = "Only motion"
    templates = [""]
    optional_input_modality = []
    input_modality: List = []
    output_modality: List = [Motion]
