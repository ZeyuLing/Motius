from typing import List
from ...modality import Audio, Caption, Motion, NumPerson
from ..base_task import BaseTask


class Speech2Gesture(BaseTask):
    abbr = "s2g"
    description = (
        "generate gesture motion from speech, with optional duration and caption"
    )
    templates = [
        "Given the speech, generate the corresponding gesture motion.",
        "Generate the gesture motion for the given speech.",
        "Create gesture movements matching the speech audio",
        "Produce body motions aligned with spoken audio",
        "Generate synchronized gestures for the audio input",
        "Develop motion patterns that follow speech rhythms",
        "Formulate gestures corresponding to vocal patterns",
        "Build body language matching the speech recording",
        "Make physical movements timed with audio speech",
        "Design kinetic responses to spoken content",
        "Construct gesture sequences from voice recordings",
        "Generate body animations synced with speech",
        "Create motion flows based on vocal audio",
        "Produce natural gestures for the given speech",
        "Develop physical expressions matching audio",
        "Form coordinated gestures with speech timing",
        "Generate conversational gestures from audio",
        "Build expressive motions aligned with speech",
        "Make body movements that mirror speech patterns",
        "Create kinetic responses to spoken words",
        "Produce gesture animations for speech audio",
        "Generate rhythm-matched motions for dialogue",
        "Develop physical reactions to speech sounds",
        "Formulate motion sequences from voice input",
        "Create body language reflecting speech content",
        "Generate movement flows matching audio cues",
        "Produce gesture patterns synchronized with speech",
        "Build physical responses to spoken audio",
        "Make motions that align with vocal rhythms",
        "Create animated gestures for speech clips",
        "Generate posture changes matching speech",
        "Develop kinetic expressions from audio",
        "Form body movements corresponding to speech",
        "Produce gesture choreography for dialogue",
        "Generate motion timing matching speech flow",
        "Create physical reactions to spoken phrases",
        "Build gesture sequences from audio patterns",
        "Make body motions that follow speech pacing",
        "Generate expressive movements for audio",
        "Develop synchronized motion-speech pairs",
        "Form kinetic animations from voice recordings",
        "Produce natural body language for speech",
        "Create rhythm-aligned motions with audio",
        "Generate physical gestures matching tone",
        "Build motion responses to speech input",
        "Make movements reflecting speech emotion",
        "Develop audio-driven gesture animations",
        "Formulate physical expressions from speech",
        "Generate conversational body movements",
        "Create speech-synchronized motion sequences",
        "Produce dialogue-accompanying gestures",
        "Build vocal-pattern-matched motions",
    ]
    input_modality: List = [Audio]
    optional_input_modality = [NumPerson, Caption]
    output_modality: List = [Motion]
