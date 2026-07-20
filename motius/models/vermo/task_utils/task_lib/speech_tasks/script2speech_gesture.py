from typing import List
from ..base_task import BaseTask
from ...modality import Audio, NumPerson, Caption, Motion, SpeechScript


class SpeechScript2SpeechGesture(BaseTask):
    abbr = "ss2sg"
    description = "Generate speech and gesture sequences from speech script text"
    templates = [
        "Given a speech script text, generate the corresponding speech and gesture sequences.",
        "Here is a speaking script text, guess the speech voice and accompanying gesture sequences.",
        "Create speech-synchronized motion sequences from a given script text.",
        "Given a text script of what is being said, synthesize the spoken audio and the matching gesture motion sequence.",
        "From this spoken script, produce a speech waveform and a synchronized gesture sequence.",
        "Take the following dialogue text and generate both the voice (speech audio) and the accompanying body/hand gestures.",
        "Using the provided speech transcript, create the corresponding vocal delivery and the co-speech gestures.",
        "Generate natural-sounding speech audio and aligned gesture motions based on the following speech text.",
        "Here is a speech transcript. Predict how it would sound when spoken and how the speaker would gesture.",
        "Create a realistic speech signal and synchronized gesture animation from this script.",
        "Turn this speaking script into (1) an audio speech track and (2) a time-aligned gesture sequence.",
        "For the given utterance text, produce speech audio and the gestures that would be performed while speaking.",
        "Given this line of speech, generate the corresponding vocal performance and body/hand motion timeline.",
        "From the input text of an utterance, infer the speech prosody and generate the matching co-speech gestures.",
        "Given the text content of a talk, synthesize a talking voice and coordinated gestural movements.",
        "Produce an audio rendering of this script and generate the accompanying expressive gestures.",
        "Take the speaker's script and output: (a) speech audio, (b) synchronized gesture trajectories.",
        "Based on the transcript below, create both the spoken voice and the gesticulation sequence over time.",
        "Using this textual speech content, generate a realistic voice track along with expressive upper-body gestures.",
        "Generate a continuous speech waveform and a frame-by-frame gesture motion from the provided script.",
        "From the provided spoken content, predict the sound of the speech and how the speaker's hands/arms move.",
        "Turn this script into talking audio plus aligned co-speech gesture motions.",
        "Given the dialogue line, synthesize the speech signal and produce a temporally aligned gesture sequence.",
        "For this speech text, create synchronized vocal audio and corresponding gestural behavior.",
        "Generate speaker audio and human-like co-speech gestures given the transcription.",
        "Using this utterance text, output the speech audio and the time-synced gestures that convey it.",
        "Take this text and imagine a person saying it; generate both the voice and their concurrent gestures.",
        "Create a natural voice-over and matching communicative gestures from the script below.",
        "From the following line, produce an audio track of the speaker and the expressive body gestures that go with it.",
        "Generate an articulated speech waveform and the associated arm/hand motion sequence from the given script.",
        "Given a spoken script, output speech audio with aligned gesture kinematics.",
        "For the input transcript, synthesize how it is spoken and how the speaker moves their hands while speaking.",
        "From this piece of speech text, infer vocal delivery and synthesize the accompanying co-speech gestures.",
        "Transform the provided text into (1) spoken audio and (2) temporally synchronized gesture poses.",
        "Using the script below, generate a voice performance and the corresponding gesture animation timeline.",
        "Produce a realistic speech signal plus synchronized communicative gestures based on the transcript.",
        "Given the speech description, generate a waveform of the speaker talking and a matching sequence of gestures.",
        "Take the transcribed speech and create both the audio narration and the gestural motion cues.",
        "Given this talk segment, synthesize the speaker’s voice and generate the gestures they would perform.",
        "From the following utterance text, output expressive speech audio and co-expressive gestures.",
        "Create both the vocal track and the upper-body motion (gestures) aligned to the provided script.",
        "Using the input script, generate time-aligned speech audio and gesture keyframes.",
        "Given a sentence the speaker will say, produce the audio of that sentence and the gestures accompanying it.",
        "Produce synchronized speech audio and communicative hand/body motion from this textual utterance.",
        "From this dialogue text, synthesize natural speech and infer the gesture dynamics over time.",
        "Generate talking audio plus gesture trajectories conditioned on the provided speech text.",
        "Given the following line, create the sound of the speaker talking and the physical gestures that match it.",
        "Take this speech script and generate the speaker’s vocal prosody together with aligned gestures.",
        "From the script content, produce a plausible speech waveform and matching gestural motion curves.",
        "Given the transcript, output (a) speech audio and (b) co-speech gesture motion synchronized to that audio.",
        "Use the following spoken text to generate expressive voice audio and temporally aligned body gestures.",
        "Create speech audio and corresponding communicative gestures conditioned on the provided utterance.",
        "From the input speech text, generate (1) the spoken voice track, and (2) the synchronized gesture sequence.",
        "Given a spoken sentence, synthesize its audio realization and the concurrent human-like gestures.",
    ]

    input_modality: List = [SpeechScript]
    optional_input_modality = [NumPerson, Caption]
    output_modality: List = [Motion, Audio]
