from typing import List
from ...modality import FutureMusic, Genre, Motion, PastMusic
from ..base_task import BaseTask


class Dance2MusicWithInit(BaseTask):
    abbr = "d2m_ar"
    description = (
        "generate music from dance motion given an initial music segment, "
        "enabling autoregressive arbitrary-length dance-to-music"
    )
    templates = [
        "Continue the music for the given dance, starting from the provided initial music segment.",
        "Given the starting music and the dance motion, generate the continuation of the music.",
        "Extend the music to match the dance, using the initial music segment as a starting point.",
        "Based on the initial music and the dance, produce the following music sequence.",
        "Continue composing music for the dance from where the initial segment left off.",
        "Generate the next music segment that follows the provided starting music and dance.",
        "Given the initial music segment and the dance motion, create the subsequent music.",
        "Produce music continuing from the initial segment, synchronized with the dance.",
        "Using the starting music and the dance, generate the rest of the soundtrack.",
        "From the given initial music, compose music that matches the dance motion.",
        "Extend the music based on the initial segment and the provided dance.",
        "Continue the composition from the starting music, following the dance.",
        "Generate music continuation for the dance, beginning from the initial music segment.",
        "Create the following music that matches the dance, starting from the given segment.",
        "Produce the rest of the music that matches the dance, given the starting segment.",
        "Starting from the initial music, generate music synchronized with the dance.",
        "Using the provided initial music and dance, create the continuation of the soundtrack.",
        "Given the starting music segment and the dance moves, generate the following music.",
        "Continue the music from the initial segment, matching the rhythm of the dance.",
        "Generate subsequent music from the initial segment, guided by the dance motion.",
        "Extend the musical piece from the starting segment to match the provided dance.",
        "From the initial music segment, produce music that follows the dance motion.",
        "Create music continuation from the given starting segment, synchronized with the dance.",
        "Using the initial music as a starting point, generate music for the dance.",
        "Produce the following music sequence starting from the provided segment and dance.",
        "Continue composing music from the starting segment to match the dance motion.",
        "Generate the next segment of the music, starting from the initial segment and dance.",
        "Given the starting music and dance, create the rest of the musical accompaniment.",
        "Extend the music from the initial segment, following the dance's rhythm.",
        "Starting from the given music, produce music that complements the dance.",
    ]
    input_modality: List = [Motion, PastMusic]
    optional_input_modality = [Genre]
    output_modality: List = [FutureMusic]
