"""Bone-name matching shared by the host API and Blender subprocess."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


SMPL22_BONE_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)


AUTO_RIG_ALIASES = {
    "Pelvis": ("pelvis", "hips", "hip", "rootjoint"),
    "L_Hip": ("lhip", "lefthip", "leftupleg", "leftthigh", "thighl", "upperlegl"),
    "R_Hip": ("rhip", "righthip", "rightupleg", "rightthigh", "thighr", "upperlegr"),
    "Spine1": ("spine", "spine0", "spine01", "lowerback"),
    "L_Knee": ("lknee", "leftknee", "leftleg", "leftlowerleg", "calfl", "shinl"),
    "R_Knee": ("rknee", "rightknee", "rightleg", "rightlowerleg", "calfr", "shinr"),
    "Spine2": ("spine1", "spine02", "midspine", "chest"),
    "L_Ankle": ("lankle", "leftankle", "leftfoot", "footl"),
    "R_Ankle": ("rankle", "rightankle", "rightfoot", "footr"),
    "Spine3": ("spine2", "spine03", "upperchest", "upperbody"),
    "L_Foot": ("lfoot", "lefttoe", "lefttoebase", "toel", "balll"),
    "R_Foot": ("rfoot", "righttoe", "righttoebase", "toer", "ballr"),
    "Neck": ("neck", "neck1"),
    "L_Collar": ("lcollar", "leftcollar", "leftshoulder", "claviclel", "shoulderl"),
    "R_Collar": ("rcollar", "rightcollar", "rightshoulder", "clavicler", "shoulderr"),
    "Head": ("head", "head1"),
    "L_Shoulder": ("lshoulder", "leftarm", "leftupperarm", "upperarml"),
    "R_Shoulder": ("rshoulder", "rightarm", "rightupperarm", "upperarmr"),
    "L_Elbow": ("lelbow", "leftelbow", "leftforearm", "lowerarml", "forearml"),
    "R_Elbow": ("relbow", "rightelbow", "rightforearm", "lowerarmr", "forearmr"),
    "L_Wrist": ("lwrist", "leftwrist", "lefthand", "handl"),
    "R_Wrist": ("rwrist", "rightwrist", "righthand", "handr"),
}


def normalize_bone_name(name: str) -> str:
    """Normalize namespaces and common rig prefixes for deterministic matching."""

    value = str(name).rsplit("|", 1)[-1].rsplit(":", 1)[-1].casefold()
    value = re.sub(r"[^a-z0-9]+", "", value)
    for prefix in ("mixamorig", "bip001", "bip01", "def", "org"):
        if value.startswith(prefix) and len(value) > len(prefix):
            value = value[len(prefix) :]
            break
    return value


def resolve_bone_map(
    target_bone_names: Iterable[str],
    supplied: Mapping[str, str] | None = None,
    *,
    strict: bool = True,
) -> dict[str, str]:
    """Resolve SMPL-22 names to a target armature's bone names."""

    target_names = tuple(str(name) for name in target_bone_names)
    exact = set(target_names)
    normalized: dict[str, list[str]] = {}
    for name in target_names:
        normalized.setdefault(normalize_bone_name(name), []).append(name)

    supplied = dict(supplied or {})
    unknown_sources = sorted(set(supplied).difference(SMPL22_BONE_NAMES))
    if unknown_sources:
        raise ValueError(f"Unknown SMPL bone names in bone_map: {unknown_sources}.")
    missing_targets = sorted(value for value in supplied.values() if value not in exact)
    if missing_targets:
        raise ValueError(f"bone_map target bones do not exist: {missing_targets}.")

    result: dict[str, str] = {}
    for source in SMPL22_BONE_NAMES:
        if source in supplied:
            result[source] = supplied[source]
            continue
        if source in exact:
            result[source] = source
            continue
        matched = False
        for alias in (*AUTO_RIG_ALIASES[source], source):
            candidates = list(
                dict.fromkeys(normalized.get(normalize_bone_name(alias), ()))
            )
            if len(candidates) == 1:
                result[source] = candidates[0]
                matched = True
                break
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous target bones for {source}: {candidates}; "
                    "provide bone_map explicitly."
                )
        if matched:
            continue

    duplicates = sorted(
        target for target in set(result.values()) if list(result.values()).count(target) > 1
    )
    if duplicates:
        raise ValueError(f"Multiple SMPL bones map to the same target bone: {duplicates}.")
    if strict:
        missing = [name for name in SMPL22_BONE_NAMES if name not in result]
        if missing:
            raise ValueError(
                "Target armature is missing required SMPL body mappings: "
                f"{missing}. Supply bone_map or set strict_bone_map=False."
            )
    return result


__all__ = [
    "AUTO_RIG_ALIASES",
    "SMPL22_BONE_NAMES",
    "normalize_bone_name",
    "resolve_bone_map",
]
