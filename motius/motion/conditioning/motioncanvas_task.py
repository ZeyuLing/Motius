"""Task Instruction Modulation for HyMotion M2M.

Maps mask strategies (M1-M7) and task modes to natural language instructions
that are CLIP-encoded and injected into the timestep embedding to provide
explicit task awareness to the model during training and inference.

Reference: MotionLab uses similar CLIP-encoded task instructions for each task.
This module provides the data flow for M2M.
"""

from __future__ import annotations

from typing import Dict, Optional

# Mask strategy → Task instruction mapping
# These instructions will be CLIP-encoded and added to the adapter signal
STRATEGY_TO_INSTRUCTION: Dict[str, str] = {
    'm1_random_cell': (
        'complete motion from sparse random cells'
    ),
    'm2_random_block': (
        'inpaint motion in random blocks'
    ),
    'm3_temporal_contiguous': (
        'extend or bridge motion temporally'
    ),
    'm4_joint_contiguous': (
        'edit specific joints or body parts'
    ),
    'm5_full_mask': (
        'generate entire motion from scratch'
    ),
    'm6_keyframe_sparse': (
        'inpaint motion between keyframes'
    ),
    'm7_scattered_joint': (
        'repair scattered joint artifacts'
    ),
}

# Default text-to-motion (M5 variant)
DEFAULT_T2M_INSTRUCTION = 'generate entire motion from scratch'

# Null instruction for unconditional generation (CFG)
NULL_INSTRUCTION = 'null'


def get_task_instruction(
    strategy: str,
    instruction_override: Optional[str] = None,
) -> str:
    """Get task instruction for a given strategy.

    Parameters
    ----------
    strategy : str
        Strategy name (e.g., 'm1_random_cell', 'm5_full_mask', or 't2m' for text-to-motion).
    instruction_override : str, optional
        If provided, use this instruction instead of the default for the strategy.

    Returns
    -------
    str
        Natural language task instruction.
    """
    if instruction_override:
        return instruction_override

    if strategy in STRATEGY_TO_INSTRUCTION:
        return STRATEGY_TO_INSTRUCTION[strategy]

    # Handle T2M (pure text-to-motion, all dims masked)
    if strategy in ('t2m', 'm5_full_mask'):
        return DEFAULT_T2M_INSTRUCTION

    # Handle null (for CFG)
    if strategy in ('null', ''):
        return NULL_INSTRUCTION

    # Fallback: warn but provide reasonable default
    return DEFAULT_T2M_INSTRUCTION


def strategy_from_mask_ratio(mask_ratio: float) -> str:
    """Infer strategy from mask ratio (for logging/analysis purposes).

    Not used in training, just for interpretation.
    """
    if mask_ratio > 0.95:
        return 'm5_full_mask'
    elif mask_ratio > 0.7:
        return 'm3_temporal_contiguous'
    elif mask_ratio > 0.4:
        return 'm1_random_cell'
    else:
        return 'm6_keyframe_sparse'
