from typing import List
from ...modality import Genre, Motion, Music, NumPerson
from ..base_task import BaseTask


class Dance2Music(BaseTask):
    abbr = "d2m"
    description = "generate music from dance motion, with optional duration and caption"
    templates = [
        "Create music that matches the dance motion.",
        "Generate a soundtrack for the provided dance.",
        "Produce music that fits the dance sequence.",
        "Compose a song that aligns with the dance moves.",
        "Make music that complements the dance motion.",
        "Generate background music for the dance.",
        "Create a musical piece that syncs with the dance.",
        "Produce audio that matches the dance rhythm.",
        "Compose a track that fits the dance style.",
        "Generate music that enhances the dance performance.",
        "Create a beat that matches the dance flow.",
        "Produce a melody that aligns with the dance steps.",
        "Compose music that reflects the dance motion.",
        "Generate a tune that fits the dance sequence.",
        "Create a rhythm that matches the dance moves.",
        "Produce a score that complements the dance.",
        "Compose music that captures the essence of the dance.",
        "Generate a soundtrack that syncs with the dance motion.",
        "Create a musical accompaniment for the dance.",
        "Produce a track that matches the dance tempo.",
        "Compose music that fits the dance's energy.",
        "Generate a melody that aligns with the dance rhythm.",
        "Create a beat that complements the dance style.",
        "Produce a song that matches the dance sequence.",
        "Compose music that enhances the dance's mood.",
        "Generate a tune that fits the dance's flow.",
        "Create a rhythm that syncs with the dance motion.",
        "Produce a score that matches the dance's tempo.",
        "Compose music that reflects the dance's energy.",
        "Generate a soundtrack that complements the dance moves.",
        "Create a musical piece that aligns with the dance rhythm.",
        "Produce a track that fits the dance's style.",
        "Compose music that captures the dance's essence.",
        "Generate a melody that matches the dance sequence.",
        "Create a beat that syncs with the dance motion.",
        "Produce a song that complements the dance's mood.",
        "Compose music that enhances the dance's energy.",
        "Generate a tune that fits the dance's rhythm.",
        "Create a rhythm that aligns with the dance style.",
        "Produce a score that matches the dance's flow.",
        "Compose music that reflects the dance's tempo.",
        "Generate a soundtrack that syncs with the dance sequence.",
        "Create a musical accompaniment that fits the dance motion.",
        "Produce a track that complements the dance's rhythm.",
        "Compose music that captures the dance's style.",
        "Generate a melody that matches the dance's energy.",
        "Create a beat that aligns with the dance's mood.",
        "Produce a song that fits the dance's flow.",
        "Compose music that enhances the dance's rhythm.",
        "Generate a tune that syncs with the dance's tempo.",
    ]
    input_modality: List = [Motion]
    optional_input_modality = [Genre, NumPerson]
    output_modality: List = [Music]
