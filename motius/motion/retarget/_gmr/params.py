"""Vendored GMR paths (SMPL-X -> Unitree G1 only).

This is a trimmed copy of ``general_motion_retargeting/params.py`` from the
vendored GMR project, reduced to the single SMPL-X -> G1 retarget path that the
``motius`` library exposes. Asset / IK-config roots are resolved relative to
this package so the library does not require an external source checkout.
"""
import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
}

IK_CONFIG_DICT = {
    "smplx": {
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
    },
}
