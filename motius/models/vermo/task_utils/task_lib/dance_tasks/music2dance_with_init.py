from typing import List
from ...modality import Caption, FutureMotion, Genre, Music, PastMotion
from ..base_task import BaseTask


class Music2DanceWithInit(BaseTask):
    abbr = "m2d_ar"
    description = (
        "generate dance motion from music given initial motion frame(s), "
        "enabling autoregressive arbitrary-length music-to-dance"
    )
    templates = [
        "Continue the dance to the given music, starting from the provided initial motion.",
        "Given the starting pose and music, generate the continuation of the dance.",
        "Extend the dance moves to match the music, using the initial motion as a starting point.",
        "Based on the initial pose and the music, produce the following dance sequence.",
        "Continue dancing to the music from where the initial motion left off.",
        "Generate the next dance segment that follows the provided starting pose and music.",
        "Given the initial body pose and the music track, create the subsequent dance motion.",
        "Produce dance movements continuing from the initial frame, synchronized with the music.",
        "Using the starting motion and the music, generate the rest of the dance.",
        "From the given initial pose, choreograph dance moves that match the music.",
        "Extend the dance sequence based on the initial motion and the provided music.",
        "Continue the choreography from the starting pose, following the music.",
        "Generate dance continuation for the music, beginning from the initial motion frame.",
        "Create the following dance moves starting from the given pose and music.",
        "Produce the rest of the dance motion that matches the music, given the starting pose.",
        "Starting from the initial motion, generate dance movements synchronized with the music.",
        "Using the provided initial frame and music, create the continuation of the dance.",
        "Given the starting body position and the track, generate the following dance moves.",
        "Continue the dance from the initial pose, matching the rhythm of the music.",
        "Generate subsequent dance movements from the initial frame, guided by the music.",
        "Extend the dance performance from the starting pose to match the provided music.",
        "From the initial body pose, produce dance moves that follow the music.",
        "Create dance continuation from the given starting motion, synchronized with the music.",
        "Using the initial pose as a starting point, generate dance moves for the music.",
        "Produce the following dance sequence starting from the provided pose and music track.",
        "Continue choreographing dance moves from the starting motion to match the music.",
        "Generate the next segment of the dance, starting from the initial pose and music.",
        "Given the starting frame and music, create the rest of the dance performance.",
        "Extend the dance from the initial motion frame, following the music's rhythm.",
        "Starting from the given pose, produce dance movements that complement the music.",
    ]
    input_modality: List = [Music, PastMotion]
    optional_input_modality = [Caption, Genre]
    output_modality: List = [FutureMotion]
