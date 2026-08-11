from typing import List
from ...modality import Audio, Caption, FutureMotion, PastMotion
from ..base_task import BaseTask


class Speech2GestureWithInit(BaseTask):
    abbr = "s2g_ar"
    description = (
        "generate gesture motion from speech given initial motion frame(s), "
        "enabling autoregressive arbitrary-length speech-to-gesture"
    )
    templates = [
        "Continue the gesture motion for the given speech, starting from the provided initial pose.",
        "Given the starting pose and speech audio, generate the continuation of the gestures.",
        "Extend the gesture movements to match the speech, using the initial motion as a starting point.",
        "Based on the initial pose and the speech, produce the following gesture sequence.",
        "Continue gesturing from where the initial motion left off, following the speech.",
        "Generate the next gesture segment that follows the provided starting pose and speech.",
        "Given the initial body pose and the speech audio, create the subsequent gesture motion.",
        "Produce gesture movements continuing from the initial frame, synchronized with the speech.",
        "Using the starting motion and the speech, generate the rest of the gestures.",
        "From the given initial pose, create gesture movements that match the speech.",
        "Extend the gesture sequence based on the initial motion and the provided speech.",
        "Continue the gestures from the starting pose, following the speech audio.",
        "Generate gesture continuation for the speech, beginning from the initial motion frame.",
        "Create the following gesture moves starting from the given pose and speech.",
        "Produce the rest of the gesture motion that matches the speech, given the starting pose.",
        "Starting from the initial motion, generate gestures synchronized with the speech.",
        "Using the provided initial frame and speech, create the continuation of the gestures.",
        "Given the starting body position and the speech, generate the following gesture moves.",
        "Continue gesturing from the initial pose, matching the rhythm of the speech.",
        "Generate subsequent gesture movements from the initial frame, guided by the speech.",
        "Extend the gesture performance from the starting pose to match the provided speech.",
        "From the initial body pose, produce gesture moves that follow the speech.",
        "Create gesture continuation from the given starting motion, synchronized with the speech.",
        "Using the initial pose as a starting point, generate gestures for the speech audio.",
        "Produce the following gesture sequence starting from the provided pose and speech.",
        "Continue generating gestures from the starting motion to match the speech.",
        "Generate the next segment of gestures, starting from the initial pose and speech.",
        "Given the starting frame and speech, create the rest of the gesture performance.",
        "Extend the gestures from the initial motion frame, following the speech's rhythm.",
        "Starting from the given pose, produce gesture movements that complement the speech.",
    ]
    input_modality: List = [Audio, PastMotion]
    optional_input_modality = [Caption]
    output_modality: List = [FutureMotion]
