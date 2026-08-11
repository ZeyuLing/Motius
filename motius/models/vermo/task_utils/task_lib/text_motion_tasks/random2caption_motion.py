from typing import List
from ...modality import Caption, Duration, Motion, NumPerson
from ..base_task import BaseTask


class Random2CaptionMotion(BaseTask):
    abbr = "n2tm"
    description = "random to text motion"
    templates = [
        "Randomly create a sequence of actions and provide a description.",
        "Generate a random set of movements and describe them in detail.",
        "Produce an arbitrary series of gestures and explain what happens.",
        "Show me a random flow of actions and describe how they unfold.",
        "Create a random pattern of motions and give a detailed account.",
        "Generate a random combination of movements and describe their sequence.",
        "Produce a random series of dynamic actions and explain their flow.",
        "Display a random set of gestures and describe their execution.",
        "Randomly generate a sequence of physical actions and describe them.",
        "Create a random arrangement of movements and provide a description.",
        "Generate a random series of kinetic motions and explain how they work.",
        "Show me a random set of body movements and describe their progression.",
        "Produce a random flow of dynamic gestures and describe their sequence.",
        "Randomly create a series of actions and explain what they entail.",
        "Generate a random pattern of physical motions and describe their flow.",
        "Display a random sequence of actions and provide a detailed explanation.",
        "Create a random set of dynamic movements and describe how they unfold.",
        "Generate a random series of gestures and explain their execution.",
        "Produce a random arrangement of actions and describe their progression.",
        "Show me a random flow of physical motions and explain their sequence.",
        "Randomly generate a set of kinetic actions and describe their flow.",
        "Create a random series of body movements and provide a detailed account.",
        "Generate a random pattern of dynamic gestures and describe their execution.",
        "Produce a random sequence of actions and explain how they unfold.",
        "Display a random set of physical motions and describe their progression.",
        "Randomly create a flow of movements and provide a detailed description.",
        "Generate a random arrangement of gestures and explain their sequence.",
        "Show me a random series of dynamic actions and describe their flow.",
        "Produce a random set of kinetic motions and explain their execution.",
        "Create a random sequence of body movements and describe their progression.",
        "Generate a random pattern of actions and provide a detailed explanation.",
        "Display a random flow of physical gestures and describe their sequence.",
        "Randomly generate a series of dynamic motions and explain their flow.",
        "Create a random set of kinetic actions and describe their execution.",
        "Generate a random arrangement of body movements and provide a description.",
        "Produce a random sequence of gestures and explain how they unfold.",
        "Show me a random flow of actions and describe their progression.",
        "Randomly create a series of physical motions and provide a detailed account.",
        "Generate a random pattern of dynamic movements and describe their flow.",
        "Display a random set of kinetic gestures and explain their sequence.",
        "Create a random arrangement of actions and describe their execution.",
        "Generate a random series of body motions and provide a detailed explanation.",
        "Produce a random flow of dynamic actions and describe their progression.",
        "Show me a random sequence of gestures and explain how they unfold.",
        "Randomly generate a set of physical movements and describe their flow.",
        "Create a random pattern of kinetic motions and provide a detailed account.",
        "Generate a random arrangement of dynamic gestures and describe their sequence.",
        "Display a random series of body actions and explain their execution.",
        "Produce a random flow of movements and describe their progression.",
    ]
    input_modality: List = [NumPerson]
    optional_input_modality = [Duration]
    output_modality: List = [Motion, Caption]
