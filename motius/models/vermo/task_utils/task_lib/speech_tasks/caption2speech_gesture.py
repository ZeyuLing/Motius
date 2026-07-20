from typing import List
from ...modality import Audio, Caption, Duration, Motion, NumPerson
from ..base_task import BaseTask


class Caption2SpeechGesture(BaseTask):
    abbr = "t2sg"
    description = "Generate speech and gesture motion pairs from motion caption, with optional duration"
    templates = [
        "Generate matching speech and gestures from gesture descriptions",
        "Create verbal output paired with body movements based on gesture captions",
        "Produce synchronized speech with corresponding physical gestures",
        "Develop spoken dialogue coordinated with gesture motions",
        "Synthesize vocal responses aligned with described gestures",
        "Generate conversational speech alongside matching body language",
        "Create speech patterns combined with appropriate gesture animations",
        "Produce synchronized verbal and physical expressions",
        "Formulate spoken words with corresponding kinetic movements",
        "Generate dialogue accompanied by gesture sequences",
        "Create vocal responses with gesture motion synchronization",
        "Produce speech synchronized with gesture movements from captions",
        "Develop matching spoken content and physical gestures",
        "Generate co-speech gestures with corresponding verbal output",
        "Create multimodal output combining speech and gesture motions",
        "Produce verbal communication integrated with physical expressions",
        "Synthesize speech patterns timed with gesture animations",
        "Generate context-matched speech and gesture pairs",
        "Create voice output coordinated with body movement sequences",
        "Produce aligned spoken language and physical gestures",
        "Develop conversational responses with gesture animations",
        "Generate gesture-accompanied speech from descriptions",
        "Create spoken dialogue with synchronized motion patterns",
        "Produce verbal expressions and matching kinetic actions",
        "Formulate speech content aligned with gesture descriptions",
        "Generate voice responses paired with gesture motions",
        "Create multimodal expressions combining speech and movement",
        "Produce synchronized talking animations with gestures",
        "Develop natural speech flow with corresponding body language",
        "Generate context-appropriate speech-gesture combinations",
        "Create vocal narratives with accompanying physical gestures",
        "Produce speech-gesture pairs from textual descriptions",
        "Synthesize synchronized verbal and non-verbal communication",
        "Generate expressive speech with gesture animations",
        "Create interactive dialogue with gesture movements",
        "Produce coherent speech and gesture sequences",
        "Develop personality-matched speech-gesture combinations",
        "Generate emotion-aware speech with corresponding gestures",
        "Create culturally appropriate speech-motion pairs",
        "Propose verbal responses with matching gesture animations",
        "Formulate dialogue sequences with physical expressions",
        "Generate context-sensitive speech and gesture bundles",
        "Create interactive speech with gesture synchronization",
        "Produce temporally aligned speech-gesture outputs",
        "Develop semantic-aware speech and motion pairs",
        "Generate personality-driven speech with gestures",
        "Create emotion-expressive speech-gesture combos",
        "Produce conversationally appropriate motion-speech pairs",
        "Synthesize culturally relevant speech-gesture sets",
        "Generate situational speech with gesture responses",
    ]
    input_modality: List = [Caption, NumPerson]
    optional_input_modality = [Duration]
    output_modality: List = [Audio, Motion]
