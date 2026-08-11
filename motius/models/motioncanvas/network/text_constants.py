__all__ = [
    "PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION",
]


PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION_DEPRECATED = """
    Describe the human motion by examining the following aspects, please keep your answer concise and under 200 words in one paragraph:
    1. The core categories of actions involved and their semantic meanings.
    2. The specific movements of body parts indicated in the action description; if not specified, reasonable representation should be made based on the action category or the connection between actions.
    3. For consecutive actions, the sequence and transitions between actions.
    4. The character's movement trajectory, including path and direction.
    5. If the interaction, style, rhythm, emotion, speed, or intensity of the action is mentioned, please specify; otherwise, do not mention these information.
    """

PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION = """
    Summarize human motion only from the user text for representation: action categories, key body-part movements, order/transitions, trajectory/direction, posture; include style/emotion/speed only if present. Explicitly capture laterality (left/right) when mentioned; do not guess. If multiple actions are described, indicate the count of distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one concise paragraph.
    """
