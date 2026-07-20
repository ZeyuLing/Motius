from typing import List
from ...modality import Caption, Motion, NumPerson
from ..base_task import BaseTask


class Motion2Caption(BaseTask):
    abbr = "m2t"
    description = "Generate caption for multiple persons from motion"
    templates = [
        "Caption the given motion.",
        "Tell me what's happening in the motion.",
        "Describe the motion in words",
        "Explain what this movement represents",
        "Create a caption for this motion",
        "Write a description of the shown action",
        "What action is being performed here?",
        "Summarize the motion sequence in text",
        "Generate a textual explanation for the movement",
        "Provide a brief description of the motion",
        "Put the physical movement into words",
        "How would you describe this activity?",
        "Interpret the body motion as text",
        "Write a short caption for this action",
        "Explain the purpose of this movement",
        "What activity does this motion depict?",
        "Translate the physical motion into a description",
        "Give a verbal account of the movement",
        "Characterize the shown motion in text",
        "Provide context for this physical action",
        "Write an explanatory note about the motion",
        "How would you caption this movement?",
        "Define the shown motion in words",
        "Explain the meaning behind this action",
        "Create a textual representation of the motion",
        "Describe the human activity shown",
        "What does this body movement signify?",
        "Generate a descriptive label for the motion",
        "Write an interpretation of the movement",
        "Explain the nature of this physical action",
        "Provide a written summary of the motion",
        "How would you label this activity?",
        "Translate the movement into a text explanation",
        "Describe the purpose of this motion",
        "Create a textual explanation of the action",
        "What story does this movement tell?",
        "Put the body language into words",
        "Explain the context of this motion",
        "Write a descriptive phrase for the action",
        "Interpret the physical movement verbally",
        "How would you summarize this motion?",
        "Generate a brief explanation of the movement",
        "Describe the key elements of this action",
        "What message does this motion convey?",
        "Create a text version of the physical movement",
        "Explain the sequence of actions shown",
        "Write a concise description of the activity",
        "Characterize the motion in simple terms",
        "Provide a textual breakdown of the movement",
        "How would you interpret this body language?",
        "Generate a narrative for the shown motion",
        "Describe the motion's intention in words",
    ]
    optional_input_modality = []
    input_modality: List = [NumPerson, Motion]
    output_modality: List = [Caption]
