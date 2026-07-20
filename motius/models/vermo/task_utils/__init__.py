import importlib
import inspect
from typing import List

from .task_lib.base_task import BaseTask
from .modality import Modality, Text

modalities = importlib.import_module(".modality", package=__name__)
task_lib = importlib.import_module(".task_lib", package=__name__)

ALL_TASKS = [
    getattr(task_lib, name)
    for name in dir(task_lib)
    if inspect.isclass(getattr(task_lib, name))
    and issubclass(getattr(task_lib, name), BaseTask)
    and getattr(task_lib, name) != BaseTask
]


ALL_MODALS: List[Modality] = [
    getattr(modalities, name)
    for name in dir(modalities)
    if inspect.isclass(getattr(modalities, name))
    and issubclass(getattr(modalities, name), Modality)
    and getattr(modalities, name) not in [Modality, Text]
]
LOCATABLE_MODALS = [modal for modal in ALL_MODALS if modal.locatable()]
ABBR_TASK_MAPPING = {task.abbr: task for task in ALL_TASKS}
NAME_MODAL_MAPPING = {modal.name: modal for modal in ALL_MODALS}


def abbr_list_to_task_list(abbr: List[str]) -> List[BaseTask]:
    if isinstance(abbr, str):
        abbr = [abbr]
    assert isinstance(abbr, list), f"input list please, got {abbr}"
    return [ABBR_TASK_MAPPING[a] for a in abbr]
