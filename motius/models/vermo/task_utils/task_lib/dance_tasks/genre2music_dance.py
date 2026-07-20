from typing import List
from ...modality import Caption, Duration, Motion, Music, NumPerson, Genre
from ..base_task import BaseTask


class Genre2MusicDance(BaseTask):
    abbr = "g2md"
    description = "generate music and dance motion w.r.t the given genre, with optional duration and caption"
    templates = [
        "Make music and dance moves that fit this genre.",
        "Create matching music and dance for this genre.",
        "Generate a song and dance based on the genre given.",
        "Come up with music and movements that match the genre.",
        "Turn this dance genre into music and cool moves.",
        "Make beats and dance steps that go with this genre.",
        "Create music and dance that brings this genre to life.",
        "From this genre, make a song and dance routine.",
        "Build music and dance moves that follow the genre.",
        "Design a dance and soundtrack for this scenario.",
        "Make up music and body movements that fit the genre.",
        "Generate background music and dance for this genre.",
        "Create a fun dance and music combo from the genre.",
        "Turn the genre into matching music and dance steps.",
        "Make dance moves and a song that match the genre.",
        "Put together music and a dance that fits this genre.",
        "From the genre, create matching tunes and moves.",
        "Build a dance routine and music track from the genre.",
        "Make music and dance that shows what the genre says.",
        "Create song and dance partners for this genre.",
        "Generate cool moves and beats based on the genre.",
        "Make a dance performance with music that fits the genre.",
        "Turn the genre into a full dance and song package.",
        "Cook up some music and dance that matches the genre.",
        "Compose music and choreograph dance moves for this genre.",
        "Develop a musical piece and dance routine based on the genre.",
        "Craft a song and dance sequence that embodies this genre.",
        "Produce music and movement that captures the essence of this genre.",
        "Design a musical composition and dance performance for the genre.",
        "Create a harmonious blend of music and dance for this genre.",
        "Generate a musical score and choreography that represents this genre.",
        "Develop a dance routine and musical accompaniment for the genre.",
        "Compose a soundtrack and design dance moves for this genre.",
        "Create a musical piece and dance sequence that fits this genre.",
        "Generate a dance performance and music that aligns with this genre.",
        "Design a dance routine and musical composition for the genre.",
        "Produce a song and dance that reflects the characteristics of this genre.",
        "Craft a musical arrangement and choreography for this genre.",
        "Create a dance and music combination that suits this genre.",
        "Generate a musical piece and dance moves that match this genre.",
        "Develop a dance routine and musical score for the genre.",
        "Compose a song and design dance steps for this genre.",
        "Create a dance performance and music that fits this genre.",
        "Design a musical composition and choreography for the genre.",
        "Produce a dance and music duo that represents this genre.",
        "Craft a musical piece and dance sequence for this genre.",
        "Generate a dance routine and musical accompaniment for the genre.",
        "Develop a song and choreography that embodies this genre.",
    ]
    input_modality: List = [Genre, NumPerson]
    optional_input_modality = [Duration, Caption]
    output_modality: List = [Music, Motion]
