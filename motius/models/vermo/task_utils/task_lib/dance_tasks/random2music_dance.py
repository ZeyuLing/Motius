from typing import List
from ...modality import Caption, Duration, Genre, Motion, Music, NumPerson
from ..base_task import BaseTask


class Random2MusicDance(BaseTask):
    abbr = "n2md"
    description = (
        "Randomly generate music and dance motion, with optional duration and caption"
    )
    templates = [
        "Create a random music-dance combination",
        "Generate spontaneous music with dance moves",
        "Produce arbitrary dance and music together",
        "Make up random dance paired with music",
        "Generate unpredictable music and movement",
        "Create chance-based dance and soundtrack",
        "Produce random dance-music synchronization",
        "Generate arbitrary rhythm with matching moves",
        "Create unexpected dance and music pairing",
        "Make random music with corresponding dance",
        "Generate improvised dance and background music",
        "Create spontaneous movement-music pairing",
        "Produce random choreography with soundtrack",
        "Generate arbitrary dance with matching beats",
        "Create haphazard music and dance sequence",
        "Make random rhythm and movement combo",
        "Generate unplanned dance-music combination",
        "Create casual music with dance motions",
        "Produce arbitrary beat-movement pairing",
        "Generate random musical dance sequence",
        "Create unexpected music-motion pairing",
        "Make up arbitrary dance with background music",
        "Generate random groove and melody combo",
        "Create chance-driven music and choreography",
        "Produce spontaneous dance-music duo",
        "Generate arbitrary tempo with matching steps",
        "Create random beat-body movement set",
        "Make unpredictable music-dance pairing",
        "Generate haphazard melody with motions",
        "Create random sound-movement combination",
        "Produce arbitrary musical choreography",
        "Generate spontaneous rhythm and dance",
        "Create random audio-kinetic pairing",
        "Make arbitrary music with dance routine",
        "Generate unexpected sound-motion combo",
        "Create random melodic movement sequence",
        "Produce chance-based dance with music",
        "Generate arbitrary harmony with motions",
        "Create spontaneous audio-choreography pair",
        "Make random musical body movements",
        "Generate unpredictable dance-sound match",
        "Create arbitrary rhythm-movement pairing",
        "Produce random music-driven choreography",
        "Generate chance-created dance and tune",
        "Create spontaneous beat-motion combo",
        "Make arbitrary musical dance routine",
        "Generate random audio-kinetic sequence",
        "Create unexpected music-movement pair",
        "Produce haphazard dance with soundtrack",
        "Generate arbitrary melody-body motion set",
    ]
    input_modality: List = [NumPerson]
    optional_input_modality = [Duration, Genre, Caption]
    output_modality: List = [Music, Motion]
