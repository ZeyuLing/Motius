from typing import List

from ...modality import Caption, Duration, Motion, NumPerson
from ..base_task import BaseTask


class Caption2Motion(BaseTask):
    abbr = "t2m"
    description = "Generate motion for both single- or multi-persons from motion caption, with optional duration"
    templates = [
        "Create motion from the given description",
        "Generate movement based on the text input",
        "Produce actions matching the provided caption",
        "Build motion sequences from text descriptions",
        "Make body movements that fit the caption",
        "Generate kinetic patterns using the description",
        "Create physical motions aligned with the text",
        "Produce movement sequences from captions",
        "Generate corresponding motions for the text",
        "Build actions that match the description",
        "Make movement flows based on the caption",
        "Create motion patterns from written prompts",
        "Generate body language matching the text",
        "Produce kinetic responses to descriptions",
        "Build physical actions from the caption",
        "Create movement that reflects the description",
        "Generate motion aligned with the text input",
        "Make actions corresponding to the caption",
        "Produce body motions based on text",
        "Create kinetic sequences from descriptions",
        "Generate movement matching the prompt",
        "Build physical patterns from the caption",
        "Make motion flows that fit the text",
        "Produce actions guided by descriptions",
        "Create body movements from text input",
        "Generate kinetic expressions using captions",
        "Make motion sequences matching the text",
        "Build movements based on descriptions",
        "Produce physical responses to captions",
        "Create action patterns from text prompts",
        "Generate body motions that follow the caption",
        "Make kinetic flows aligned with the text",
        "Produce movement matching the description",
        "Build physical actions using the caption",
        "Create motion expressions from text input",
        "Generate body patterns based on descriptions",
        "Make kinetic sequences that fit the caption",
        "Produce movement aligned with the text",
        "Create actions corresponding to the description",
        "Generate physical motions from captions",
        "Build body movements matching the text",
        "Make kinetic responses using descriptions",
        "Produce motion flows based on captions",
        "Create movement sequences from text prompts",
        "Generate actions that reflect the description",
        "Make physical patterns guided by text",
        "Produce body motions aligned with captions",
        "Create kinetic expressions from descriptions",
        "Generate movement that follows the text",
        "Build motions matching the caption input",
    ]
    optional_input_modality = [Duration]
    input_modality: List = [Caption, NumPerson]
    output_modality: List = [Motion]
