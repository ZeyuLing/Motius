from typing import List
from ...modality import Audio, Motion, NumPerson
from ..base_task import BaseTask


class Gesture2Speech(BaseTask):
    abbr = "g2s"
    description = (
        "generate speech from gesture motion, with optional duration and caption"
    )
    templates = [
        "Given the gesture during speech, guess what the speaker(s) is(are) saying.",
        "Predict the speech content based on the accompanying gestures.",
        "From the observed body movements, infer the spoken words.",
        "Determine what is being said by analyzing the speaker's gestures.",
        "Interpret the speech by understanding the corresponding hand motions.",
        "Decode the verbal message from the nonverbal gestures.",
        "Translate the body language into spoken words.",
        "Reconstruct the speech from the accompanying hand movements.",
        "Estimate the verbal content based on gesture patterns.",
        "Generate speech that matches the observed gestures.",
        "What words would correspond to these body movements?",
        "Match the appropriate speech to these gestures.",
        "Synthesize speech that aligns with these hand motions.",
        "Infer the verbal communication from these physical cues.",
        "Produce speech that complements these body gestures.",
        "Determine the likely spoken words from these movements.",
        "What would someone be saying while making these gestures?",
        "Create a verbal message that matches these hand motions.",
        "Predict the spoken words that accompany these gestures.",
        "Generate appropriate speech for these body movements.",
        "What speech would naturally follow these gestures?",
        "Recreate the verbal content from these physical expressions.",
        "Synthesize the corresponding speech for these motions.",
        "Infer the likely verbal communication from these gestures.",
        "Produce speech that would typically accompany these movements.",
        "Determine the spoken words that match these hand gestures.",
        "What verbal message would complement these body motions?",
        "Generate speech that corresponds to these physical expressions.",
        "Predict the verbal content based on these gesture patterns.",
        "Reconstruct the spoken words from these body movements.",
        "What would someone be saying while performing these gestures?",
        "Create speech that aligns with these observed motions.",
        "Determine the appropriate verbal message for these gestures.",
        "Generate speech that matches the intensity and style of these motions.",
        "Infer the spoken words from these expressive hand movements.",
        "Produce speech that would naturally follow these body gestures.",
        "What verbal communication would accompany these motions?",
        "Recreate the likely speech from these physical expressions.",
        "Synthesize speech that corresponds to these gesture patterns.",
        "Predict the verbal message that matches these body movements.",
        "Generate appropriate speech for these expressive gestures.",
        "Determine what would be said during these hand motions.",
        "What speech would typically accompany these body movements?",
        "Create verbal content that matches these physical expressions.",
        "Infer the likely spoken words from these gesture patterns.",
        "Produce speech that aligns with these observed body motions.",
        "Reconstruct the verbal message from these hand gestures.",
        "What would someone be saying while making these specific motions?",
        "Generate speech that corresponds to these expressive movements.",
        "Determine the appropriate verbal content for these physical gestures.",
    ]
    input_modality: List = [Motion]
    optional_input_modality = [NumPerson]
    output_modality: List = [Audio]
