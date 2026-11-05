# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Predicted foot contacts stored with generated motion.

Each frame holds four foot contacts, in the order left heel, left toe,
right heel, right toe. Each value is a confidence from 0 to 1 that the heel
or toe is planted. Raw model values can fall outside this range.
The engines clamp them to this range.
"""

import warp as wp

from motius.motion.runtime.motionbricks.time_utils import frame_index_from_time

FOOT_CONTACT_COUNT = 4


@wp.kernel
def _blend_foot_contacts_kernel(
    foot_contacts: wp.array2d(dtype=wp.float32),
    frame: wp.int32,
    next_frame: wp.int32,
    blend: wp.float32,
    output: wp.array(dtype=wp.float32),
):
    contact = wp.tid()
    output[contact] = wp.lerp(
        foot_contacts[frame, contact], foot_contacts[next_frame, contact], blend
    )


def validate_foot_contacts(
    foot_contacts: wp.array | None, capacity: int, device
) -> None:
    """Check that foot_contacts is None or a float32 [capacity, 4] array on device.

    Anything else raises ValueError.

    Args:
        foot_contacts: Array to check, or None.
        capacity: Number of frames the array must hold.
        device: Device the array must be on.
    """
    if foot_contacts is None:
        return
    if foot_contacts.dtype != wp.float32 or foot_contacts.shape != (
        capacity,
        FOOT_CONTACT_COUNT,
    ):
        raise ValueError(
            f"foot_contacts must be float32 with shape [{capacity}, {FOOT_CONTACT_COUNT}]."
        )
    if foot_contacts.device != device:
        raise ValueError("foot_contacts must be on the same device as the poses.")


def sample_foot_contacts(
    buffer, time: float, output: wp.array, stream: wp.Stream | None = None
) -> None:
    """Write a buffer's foot contacts sampled at time into output.

    The two nearest frames are blended.

    Args:
        buffer: Animation buffer that has foot contacts.
        time: Sample time in seconds.
        output: Float32 array that receives the four foot contacts, on the buffer's
            device.
        stream: Stream for the work. None uses the current stream.
    """
    if buffer.foot_contacts is None:
        raise ValueError(
            "Cannot sample foot contacts from an animation buffer that has none."
        )
    if buffer.num_frames == 0:
        raise ValueError("Cannot sample an empty animation buffer.")
    if output.dtype != wp.float32 or output.shape != (FOOT_CONTACT_COUNT,):
        raise ValueError(
            f"output must be a float32 array of {FOOT_CONTACT_COUNT} foot contacts."
        )
    if output.device != buffer.foot_contacts.device:
        raise ValueError("output must be on the same device as the foot contacts.")
    frame, next_frame, blend = frame_index_from_time(
        time, buffer.sample_rate, buffer.num_frames
    )
    wp.launch(
        _blend_foot_contacts_kernel,
        dim=FOOT_CONTACT_COUNT,
        inputs=[buffer.foot_contacts, frame, next_frame, blend],
        outputs=[output],
        device=output.device,
        stream=stream,
    )
