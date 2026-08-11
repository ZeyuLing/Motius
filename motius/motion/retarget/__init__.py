"""Optional motion-retargeting backends.

Heavy dependencies are imported only when a concrete retargeter is requested.
"""

__all__ = [
    "hml263_to_motion135",
    "retarget_hml263_clip",
    "GMRSMPLToG1Retargeter",
    "G1_JOINT_NAMES",
    "G1_JOINT_LIMITS",
    "GMR_Y_UP_FROM_Z_UP",
    "GMR_Z_UP_FROM_Y_UP",
    "SMPLSOMARetargeter",
    "KIMODOSOMAToSMPLRetargeter",
    "smpl_motion135_to_soma30",
    "kimodo_soma_to_smpl_motion135",
]


def __getattr__(name: str):
    if name in {"hml263_to_motion135", "retarget_hml263_clip"}:
        from . import hml263_smpl

        return getattr(hml263_smpl, name)
    if name in {
        "GMRSMPLToG1Retargeter",
        "G1_JOINT_NAMES",
        "G1_JOINT_LIMITS",
        "GMR_Y_UP_FROM_Z_UP",
        "GMR_Z_UP_FROM_Y_UP",
    }:
        from . import smpl_g1

        return getattr(smpl_g1, name)
    if name in {
        "SMPLSOMARetargeter",
        "KIMODOSOMAToSMPLRetargeter",
        "smpl_motion135_to_soma30",
        "kimodo_soma_to_smpl_motion135",
    }:
        from . import smpl_soma

        return getattr(smpl_soma, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
