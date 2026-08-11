from typing import List
from ...modality import Audio, Caption, Duration, Motion, NumPerson
from ..base_task import BaseTask


class Random2SpeechGesture(BaseTask):
    abbr = "n2sg"
    description = "Randomly generate speech audio and gesture motion pairs, with optional duration"
    templates = [
        "Make a random speech audio and the corresponding gesture motion",
        "Make a speech, provide me with the audio and time-synced gesture motion",
        "Create random speech with matching gestures",
        "Generate spontaneous voice and body movements",
        "Make up speech audio and timed gestures",
        "Produce arbitrary talking with motions",
        "Create improvised speech with gestures",
        "Generate random conversation with movements",
        "Make spontaneous dialogue and gestures",
        "Produce casual speech with body language",
        "Create unplanned talking with motions",
        "Generate impromptu speech and gestures",
        "Make up on-the-spot speech with motions",
        "Produce random vocal sounds and moves",
        "Create chance-based speech with gestures",
        "Generate unexpected speech and movements",
        "Make arbitrary talking with body actions",
        "Produce random voice clips with gestures",
        "Create haphazard speech and motions",
        "Generate offhand remarks with moves",
        "Make casual chatting with gestures",
        "Produce spontaneous speech-motion pairs",
        "Create unscripted dialogue with gestures",
        "Generate random utterances and motions",
        "Make up extemporaneous speech with moves",
        "Produce arbitrary verbalizations and gestures",
        "Create unpredictable speech with motions",
        "Generate spur-of-the-moment speech-gesture pairs",
        "Make random voice recordings with moves",
        "Produce improvised talking with body language",
        "Create ad-libbed speech and gestures",
        "Generate makeshift speech with movements",
        "Make up unrehearsed talking with motions",
        "Produce accidental speech and gestures",
        "Create incidental voice with body moves",
        "Generate unpremeditated speech-motion combos",
        "Make arbitrary verbal expressions with gestures",
        "Produce random spoken words and motions",
        "Create spontaneous narration with moves",
        "Generate casual remarks with body actions",
        "Make up unplanned speech with gestures",
        "Produce extempore talking and movements",
        "Create impulsive speech with motions",
        "Generate unexpected voice-gesture pairs",
        "Make random vocalizations with body moves",
        "Produce improvised speech-sound with gestures",
        "Create off-the-cuff speech and motions",
        "Generate accidental talking with moves",
        "Make instantaneous speech-gesture combos",
        "Produce arbitrary verbal output with motions",
        "Create momentary speech with gestures",
        "Generate unpremeditated voice and moves",
    ]
    input_modality: List = [NumPerson]
    optional_input_modality = [Duration, Caption]
    output_modality: List = [Audio, Motion]
