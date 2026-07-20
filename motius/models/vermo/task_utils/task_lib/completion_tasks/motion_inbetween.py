from typing import List
from ...modality import (
    Duration,
    FutureMotion,
    MiddleMotion,
    PastMotion,
    Motion,
    Caption,
    NumPerson,
)
from ..base_task import BaseTask


class MotionInbetween(BaseTask):
    abbr = "inbetween"
    description = "Given the motion without the middle frame, generate the middle frame"
    templates = [
        "Given the motion of past frames and future frames, generate the middle frame",
        "Fill in the missing frame between the past and future frames",
        "Interpolate the motion between the past and future frames",
        "Generate the middle frame of the motion",
        "Create the missing frame between the past and future frames",
        "Infer the motion of the middle frame",
        "Fill in the missing frame between the past and future frames",
        "Interpolate the motion between the past and future frames",
        "Generate the middle frame of the motion",
        "Create the missing frame between the past and future frames",
        "Infer the motion of the middle frame",
        "Predict the middle motion between past and future frames",
        "Bridge the gap between start and end motion segments",
        "Reconstruct the missing motion between two clips",
        "Simulate the transition between previous and next frames",
        "Determine the intermediate motion step",
        "Complete the motion sequence between time points",
        "Estimate the connecting frame between past and future",
        "Calculate the midpoint motion given context frames",
        "Derive the middle action from surrounding frames",
        "Model the motion transition between time segments",
        "Synthesize the frame that links past and future",
        "Find the logical motion between two sequences",
        "Extrapolate the middle movement from context",
        "Generate the bridging frame between clips",
        "Restore the missing transitional motion",
        "Infer the connecting action between frames",
        "Produce the intermediate motion step",
        "Compute the motion between time points",
        "Develop the transitional frame between sequences",
        "Formulate the midpoint movement in the timeline",
        "Construct the motion bridge between clips",
        "Design the missing link in motion sequence",
        "Assemble the middle frame using context",
        "Deduce the transitional action between frames",
        "Render the motion that connects two segments",
        "Form the logical transition between movements",
        "Piece together the missing motion step",
        "Fabricate the connecting frame in timeline",
        "Shape the intermediate motion between clips",
        "Formulate the motion continuity between frames",
    ]
    optional_input_modality = [Duration, Caption, NumPerson]
    input_modality: List = [PastMotion, FutureMotion]
    output_modality: List = [MiddleMotion]
