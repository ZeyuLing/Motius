from abc import ABC
from typing import List


class BaseTask(ABC):
    """Base class for all tasks."""

    abbr = None  # abbreviation of the task
    description = "description of the task"  # Just for documentation
    multi_person = False  # whether the task is for multiple persons
    templates: List[str] = (
        None  # templates for the tasks, each template is a string describing the task
    )
    optional_input_modality = (
        []
    )  # optional input modalities for the task, each will be randomly selected
    input_modality = []  # input modalities for the task
    output_modality = []  # output modalities for the task

    @classmethod
    def all_modality(cls):
        return cls.input_modality + cls.output_modality + cls.optional_input_modality

    @classmethod
    def essential_modality(cls):
        return cls.input_modality + cls.output_modality

    @classmethod
    def num_templates(cls):
        return len(cls.templates) if cls.templates is not None else 0
