from typing import List
from ..base_task import BaseTask
from ...modality import Caption, Duration, Motion, Music, NumPerson, Genre


class Caption2MusicDance(BaseTask):
    abbr = "t2md"
    description = (
        "Randomly generate music and dance motion, with optional duration and caption"
    )
    templates = [
        "Make music and dance moves that fit this description.",
        "Create matching music and dance for this caption.",
        "Generate a song and dance based on the text given.",
        "Come up with music and movements that match the description.",
        "Turn this dance caption into music and cool moves.",
        "Make beats and dance steps that go with this text.",
        "Create music and dance that brings this caption to life.",
        "From this description, make a song and dance routine.",
        "Build music and dance moves that follow the caption.",
        "Design a dance and soundtrack for this scenario.",
        "Make up music and body movements that fit the story.",
        "Generate background music and dance for this caption.",
        "Create a fun dance and music combo from the text.",
        "Turn these words into matching music and dance steps.",
        "Make dance moves and a song that match the description.",
        "Put together music and a dance that fits this caption.",
        "From the caption, create matching tunes and moves.",
        "Build a dance routine and music track from the text.",
        "Make music and dance that shows what the caption says.",
        "Create song and dance partners for this description.",
        "Generate cool moves and beats based on the text.",
        "Make a dance performance with music that fits the caption.",
        "Turn the description into a full dance and song package.",
        "Cook up some music and dance that matches the words.",
        "Create matching sound and movement for this caption.",
        "Make beats and body moves that tell the caption's story.",
        "From the text, build a complete dance and music set.",
        "Design dance steps and background music for this idea.",
        "Generate a dance video soundtrack and moves from text.",
        "Make music to dance to that fits the description.",
        "Turn these words into a dance challenge with music.",
    ]
    input_modality: List = [Caption, NumPerson]
    optional_input_modality = [Duration, Genre]
    output_modality: List = [Music, Motion]
