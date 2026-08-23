"""Optional Make-It-Animatable backend for diverse static characters.

The backend is intentionally isolated from the deterministic local template
fitter.  It uploads one user-provided character to a configured Gradio Space
and downloads the predicted Mixamo-compatible FBX.  Motius then normalizes the
rig to its canonical SMPL22 naming contract in Blender.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_MIA_SPACE = "jasongzy/Make-It-Animatable"
MIA_REST_POSES = frozenset({"T-pose", "A-pose", "No"})

MIXAMO_TO_SMPL22 = {
    "Hips": "Pelvis",
    "LeftUpLeg": "L_Hip",
    "RightUpLeg": "R_Hip",
    "Spine": "Spine1",
    "LeftLeg": "L_Knee",
    "RightLeg": "R_Knee",
    "Spine1": "Spine2",
    "LeftFoot": "L_Ankle",
    "RightFoot": "R_Ankle",
    "Spine2": "Spine3",
    "LeftToeBase": "L_Foot",
    "RightToeBase": "R_Foot",
    "Neck": "Neck",
    "LeftShoulder": "L_Collar",
    "RightShoulder": "R_Collar",
    "Head": "Head",
    "LeftArm": "L_Shoulder",
    "RightArm": "R_Shoulder",
    "LeftForeArm": "L_Elbow",
    "RightForeArm": "R_Elbow",
    "LeftHand": "L_Wrist",
    "RightHand": "R_Wrist",
}


def _returned_file(value: object) -> Path | None:
    """Extract an existing local file from a Gradio file payload."""

    if isinstance(value, Mapping):
        for key in ("path", "value"):
            if key in value:
                candidate = _returned_file(value[key])
                if candidate is not None:
                    return candidate
        return None
    if isinstance(value, (str, Path)):
        candidate = Path(value)
        return candidate if candidate.is_file() else None
    return None


def request_make_it_animatable(
    character_path: str | Path,
    output_fbx: str | Path,
    *,
    space: str = DEFAULT_MIA_SPACE,
    rest_pose: str = "No",
) -> dict[str, object]:
    """Run the public/self-hosted MIA Gradio pipeline and copy its FBX result.

    ``gradio-client`` is imported lazily so the built-in local template backend
    remains dependency-free.  The configured Space receives the character
    file; callers handling private assets should use a trusted self-hosted
    endpoint instead of the public default.
    """

    source = Path(character_path).expanduser().resolve()
    destination = Path(output_fbx).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.suffix.casefold() != ".fbx":
        raise ValueError("The raw Make-It-Animatable output must use .fbx.")
    if rest_pose not in MIA_REST_POSES:
        choices = ", ".join(sorted(MIA_REST_POSES))
        raise ValueError(f"rest_pose must be one of {choices}.")
    if not str(space).strip():
        raise ValueError("space must be a non-empty Gradio Space identifier or URL.")

    try:
        from gradio_client import Client, handle_file
    except ImportError as error:
        raise ImportError(
            "The Make-It-Animatable backend requires gradio-client. Install "
            "Motius with `pip install -e '.[auto-rig]'`."
        ) from error

    client = Client(str(space))
    result = client.predict(
        handle_file(str(source)),
        True,
        rest_pose,
        [],
        False,
        0.01,
        False,
        True,
        "LeftArm",
        True,
        None,
        False,
        True,
        api_name="/pipeline",
    )

    # The official Space exposes the final FBX as output 8.  Prefer that
    # documented position, while accepting a direct file payload from a
    # compatible self-hosted endpoint.
    payload = result
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) <= 8:
            raise RuntimeError(
                "Make-It-Animatable returned fewer outputs than its /pipeline contract."
            )
        payload = result[8]
    returned = _returned_file(payload)
    if returned is None:
        raise RuntimeError(
            "Make-It-Animatable did not return a readable FBX file payload."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(returned, destination)
    if destination.stat().st_size == 0:
        raise RuntimeError("Make-It-Animatable returned an empty FBX file.")
    return {
        "name": "Make-It-Animatable",
        "space": str(space),
        "api_name": "/pipeline",
        "rest_pose": rest_pose,
        "upstream": "https://github.com/jasongzy/Make-It-Animatable",
        "network_upload": True,
    }


__all__ = [
    "DEFAULT_MIA_SPACE",
    "MIA_REST_POSES",
    "MIXAMO_TO_SMPL22",
    "request_make_it_animatable",
]
