from typing import List
from ...modality import Caption, Duration, FutureMotion, Motion, PastMotion, NumPerson
from ..base_task import BaseTask


class MotionPrediction(BaseTask):
    abbr = "pred"
    description = "Given the past few frames of motion, predict the future"
    templates = [
        "Given the motion of past frames, predict the future motion",
        "Predict the future motion based on the past frames",
        "Forecast upcoming movements using past motion data",
        "Project future actions from previous frame sequences",
        "Anticipate coming motion based on earlier frames",
        "Extend the timeline by predicting next movements",
        "Continue the motion flow beyond current frames",
        "Simulate subsequent actions using existing motion",
        "Estimate future movement patterns from past frames",
        "Generate following motions using prior movement data",
        "Extrapolate upcoming actions from current motion",
        "Develop future motion sequences using past frames",
        "Model subsequent movements based on existing actions",
        "Calculate probable future motions from prior frames",
        "Determine what follows in the motion sequence",
        "Extend the action timeline with predicted motions",
        "Predict subsequent frames in the movement sequence",
        "Continue the action flow with projected motions",
        "Anticipate the next phase of movement",
        "Generate motion continuation from existing frames",
        "Project the movement trajectory forward",
        "Simulate the progression of current motions",
        "Forecast the motion path beyond current frames",
        "Predict how the current action will develop",
        "Extend the movement sequence into the future",
        "Calculate the logical continuation of motions",
        "Determine the natural progression of actions",
        "Model how current movements will evolve",
        "Anticipate the motion sequence's next steps",
        "Generate plausible future action sequences",
        "Predict subsequent phases of the movement",
        "Continue the kinetic pattern into future frames",
        "Project how actions will unfold over time",
        "Simulate the motion's temporal development",
        "Forecast movement evolution from current state",
        "Extend the physical action timeline",
        "Predict the kinetic chain's continuation",
        "Generate chronological motion extensions",
        "Anticipate the action sequence's progression",
        "Model temporal development of movements",
        "Calculate motion continuity probabilities",
        "Determine the action flow's next stage",
    ]
    optional_input_modality = [Duration, Caption, NumPerson]
    input_modality: List = [PastMotion]
    output_modality: List = [FutureMotion]
