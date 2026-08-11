#!/usr/bin/env python3
"""Synchronize task demos and canonical Leaderboard metrics in Model Cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from tools.normalize_model_cards import (
        MODEL_ZOO_DIR,
        REPO_ROOT,
        TASKS_END,
        TASK_REGISTRY,
        _catalog_cards,
        _catalog_task_contracts,
    )
except ModuleNotFoundError:
    from normalize_model_cards import (
        MODEL_ZOO_DIR,
        REPO_ROOT,
        TASKS_END,
        TASK_REGISTRY,
        _catalog_cards,
        _catalog_task_contracts,
    )


T2M_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_t2m_humanml3d"
    / "t2m_results.json"
)
M2T_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_m2t_humanml3d"
    / "m2t_results.json"
)
M2D_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_music_to_dance"
    / "music_to_dance_results.json"
)
D2M_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_dance_to_music"
    / "dance_to_music_results.json"
)
TEMPORAL_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_temporal_condition"
    / "temporal_control_results.json"
)
MONOCULAR_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_monocular_capture"
    / "monocular_capture_results.json"
)
MOTION_REPAIR_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_motion_repair"
    / "motion_repair_results.json"
)
MOTION_TRACKING_MUJOCO_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_motion_tracking_mujoco"
    / "motion_tracking_results.json"
)
MOTION_TRACKING_ISAACLAB_RESULTS = (
    REPO_ROOT
    / "docs"
    / "leaderboards"
    / "hf_space_motion_tracking_isaaclab"
    / "motion_tracking_results.json"
)
RELEASE_MANIFEST = MODEL_ZOO_DIR / "release_manifest.json"
VIDEO_ATTACHMENTS = MODEL_ZOO_DIR / "video_attachments.json"
MODEL_ZOO_README = MODEL_ZOO_DIR / "README.md"

DEMO_START = "<!-- MOTIUS_TASK_DEMOS:START -->"
DEMO_END = "<!-- MOTIUS_TASK_DEMOS:END -->"
FRAME_RATE_START = "<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->"
FRAME_RATE_END = "<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->"
METRICS_START = "<!-- MOTIUS_CANONICAL_METRICS:START -->"
METRICS_END = "<!-- MOTIUS_CANONICAL_METRICS:END -->"
ZOO_METRICS_START = "<!-- MOTIUS_MODEL_ZOO_METRICS:START -->"
ZOO_METRICS_END = "<!-- MOTIUS_MODEL_ZOO_METRICS:END -->"

FRAME_RATE_CONTRACTS = {
    "any2track": {
        "training": "50 Hz G1 control after reference-motion preprocessing",
        "preview": "50 Hz MuJoCo rollout; public media encoded at 30 fps",
    },
    "ardy": {
        "training": "ARDY-330: 20 fps; Unitree G1: 25 fps",
        "preview": (
            "ARDY-330: 20 fps native; G1 and control previews retain their "
            "checkpoint clock"
        ),
    },
    "bailando": {
        "training": "60 fps motion; 7.5 fps music features",
        "preview": (
            "30 fps, duration-preserving 60→30 fps motion resampling with "
            "audio synchronization"
        ),
    },
    "beyondmimic": {
        "training": "50 Hz G1 control for the official deployment recipe",
        "preview": "Canonical qpos playback at 30 fps",
    },
    "condmdi": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "dart": {
        "training": "20 fps (DART/HumanML3D motion)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "edge": {
        "training": "30 fps motion and Jukebox features",
        "preview": "30 fps native with synchronized audio",
    },
    "flowmdm": {
        "training": "HumanML3D checkpoint: 20 fps; BABEL checkpoint: 30 fps",
        "preview": (
            "T2M/temporal: 30 fps after duration-preserving 20→30 fps "
            "resampling; sequential BABEL: 30 fps native"
        ),
    },
    "gem_smpl": {
        "training": "30 fps temporal clips",
        "preview": "30 fps native model clock",
    },
    "gem_x": {
        "training": "30 fps temporal clips",
        "preview": "30 fps native model clock",
    },
    "gvhmr": {
        "training": "30 fps temporal clips",
        "preview": "30 fps native model clock",
    },
    "hymotion_t2m": {
        "training": "30 fps (HY-Motion-201 or G1-38, artifact-dependent)",
        "preview": "30 fps native",
    },
    "humanoid_gpt": {
        "training": "50 Hz G1 control; official training code is not released",
        "preview": "30 fps visualization sampled from the 50 Hz physical rollout",
    },
    "intergen": {
        "training": "30 fps (InterHuman-262)",
        "preview": "30 fps native",
    },
    "intermask": {
        "training": "30 fps (InterHuman-262)",
        "preview": "30 fps native",
    },
    "kimodo": {
        "training": "30 fps for released SOMA, SMPL-X, and G1 checkpoints",
        "preview": "30 fps native",
    },
    "maskcontrol": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "mdm": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "mld": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "mogents": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "momask": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "motionbricks": {
        "training": "30 fps (Unitree G1)",
        "preview": "30 fps native",
    },
    "motioncanvas": {
        "training": "30 fps on a 360-frame canvas",
        "preview": "30 fps native",
    },
    "motionclr": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "motiongpt": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "motiongpt3": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "motionlcm": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "motionmillion": {
        "training": "30 fps (MotionStreamer-272 / humanml3d_272)",
        "preview": "30 fps native",
    },
    "motionstreamer": {
        "training": "30 fps (MotionStreamer-272)",
        "preview": "30 fps native",
    },
    "omnicontrol": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "prism": {
        "training": "30 fps (prism_motion138)",
        "preview": "30 fps native",
    },
    "projflow": {
        "training": "20 fps (ACMDM Flow prior on HumanML3D)",
        "preview": "20 fps native",
    },
    "protomotions": {
        "training": "50 Hz G1 control with BONES-SEED reference motion",
        "preview": "30 FPS media playback; unified ONNX runtime remains 50 Hz",
    },
    "prompthmr": {
        "training": (
            "No single fixed motion clock; the image model is trained "
            "frame-wise"
        ),
        "preview": "Input-video clock unless `output_fps` is requested",
    },
    "sonic": {
        "training": "50 Hz G1 control with 20 ms reference sampling",
        "preview": "30 FPS media playback; encoder/decoder runtime remains 50 Hz",
    },
    "t2mgpt": {
        "training": "20 fps (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "tm2d": {
        "training": "60 fps motion; 7.5 fps music features",
        "preview": (
            "30 fps, duration-preserving 60→30 fps motion resampling with "
            "synchronized audio for Music-to-Dance"
        ),
    },
    "tm2t": {
        "training": "20 fps motion input (HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "unimumo": {
        "training": "60 fps motion; 50 Hz shared motion/audio code rate",
        "preview": (
            "30 fps, duration-preserving 60→30 fps motion resampling with "
            "synchronized audio for Music-to-Dance and generated audio for "
            "Dance-to-Music"
        ),
    },
    "vermo": {
        "training": "20 fps motion input (VerMo-138)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
    "vimogen": {
        "training": "20 fps (DART276 / HumanML3D)",
        "preview": "30 fps, duration-preserving 20→30 fps resampling",
    },
}

VIEWER_BASE = {
    "text_to_motion": (
        "https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/"
        "cases/index.html"
    ),
    "motion_to_text": (
        "https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/"
        "cases/index.html"
    ),
    "sequential_text_to_motion": (
        "https://zeyuling-babel-sequential-generation-leaderboard.static."
        "hf.space/cases/index.html"
    ),
    "temporal_motion_completion": (
        "https://zeyuling-temporal-condition-leaderboard.static.hf.space/"
        "cases/start_1f/index.html"
    ),
    "music_to_dance": (
        "https://zeyuling-music-to-dance-aistpp-leaderboard.static."
        "hf.space/cases/index.html"
    ),
    "dance_to_music": (
        "https://zeyuling-dance-to-music-aistpp-leaderboard.static."
        "hf.space/cases/index.html"
    ),
    "motion_repair": (
        "https://zeyuling-motion-repair-brokenamass-leaderboard.static."
        "hf.space/cases/index.html"
    ),
}

VIEWER_MANIFESTS = {
    "text_to_motion": (
        REPO_ROOT
        / "docs/leaderboards/hf_space_t2m_humanml3d/cases/manifest.json"
    ),
    "motion_to_text": (
        REPO_ROOT
        / "docs/leaderboards/hf_space_m2t_humanml3d/cases/manifest.json"
    ),
    "sequential_text_to_motion": (
        REPO_ROOT
        / "docs/leaderboards/hf_space_babel_sequential/cases/manifest.json"
    ),
    "music_to_dance": (
        REPO_ROOT
        / "docs/leaderboards/hf_space_music_to_dance/cases/manifest.json"
    ),
    "motion_repair": (
        REPO_ROOT
        / "docs/leaderboards/hf_space_motion_repair/cases/manifest.json"
    ),
}

METHOD_KEYS = {
    "hymotion_t2m": "hymotion1b",
    "motionmillion": "gotozero7b",
    "prism": "prismkafs",
}

T2M_ROW_KEYS = {
    "condmdi": {"default": ("CondMDI", "official")},
    "dart": {"default": ("DART", "official")},
    "flowmdm": {"default": ("FlowMDM", "official")},
    "hymotion_t2m": {
        "full": ("HYMotion", "1.0B · 360f"),
        "lite": ("HYMotion", "0.46B · 360f"),
    },
    "kimodo": {"default": ("KIMODO", "SMPL-X RP")},
    "maskcontrol": {"default": ("MaskControl", "official")},
    "mdm": {"default": ("MDM", "official")},
    "mld": {"default": ("MLD", "canonical")},
    "mogents": {"default": ("MoGenTS", "official")},
    "momask": {"default": ("MoMask", "official")},
    "motionclr": {"default": ("MotionCLR", "official")},
    "motiongpt": {"default": ("MotionGPT", "official")},
    "motiongpt3": {"default": ("MotionGPT3", "official")},
    "motionlcm": {"default": ("MotionLCM", "canonical")},
    "motioncanvas": {"default": ("MotionCanvas", "0.46B · 360f")},
    "motionmillion": {
        "7b train-only": ("GoToZero", "7B-train"),
        "3b train-only": ("GoToZero", "3B-train"),
    },
    "motionstreamer": {"default": ("MotionStreamer", "official")},
    "prism": {
        "prism 1.0": ("PRISM", "1.0"),
        "prism-kt + kafs": ("PRISM", "KAFS cfg5 · e20"),
    },
    "t2mgpt": {"default": ("T2M-GPT", "official")},
    "tm2d": {"default": ("TM2D", "official E0190/E0020")},
    "unimumo": {"default": ("UniMuMo", "zero-shot")},
    "vimogen": {
        "1.3b prompt-rewrite": ("ViMoGen", "1.3B prompt-rewrite"),
    },
}

RELEASE_PACKAGE_ALIASES = {
    "prism_1_0": ("prism", "PRISM 1.0"),
    "prism_kt": ("prism", "PRISM-KT + KAFS"),
}

MONOCULAR_METHODS = {
    "gem_smpl": "GEM-SMPL",
    "gem_x": "GEM-X",
    "gvhmr": "GVHMR",
    "prompthmr": "PromptHMR-Video",
}

TP2M_METHODS = {
    "flowmdm": "FlowMDM",
    "kimodo": "KIMODO",
    "motionstreamer": "MotionStreamer",
    "prism": "PRISM-KT",
}

TEMPORAL_METHOD_IDS = {
    "motioncanvas": "ours",
}

METRIC_SOURCES = {
    "text_to_motion": (
        "../leaderboards/hf_space_t2m_humanml3d/t2m_results.json",
        "HumanML3D semantic, physical, and paper rows",
    ),
    "motion_to_text": (
        "../leaderboards/hf_space_m2t_humanml3d/m2t_results.json",
        "HumanML3D M2T metrics",
    ),
    "sequential_text_to_motion": (
        "https://huggingface.co/spaces/ZeyuLing/"
        "babel-sequential-generation-leaderboard",
        "BABEL semantic and transition metrics; normalized uTMR FID",
    ),
    "text_to_multi_person_motion": (
        "../leaderboards/text_to_multi_person_interhuman.md",
        "InterHuman protocol; rows remain pending",
    ),
    "temporal_motion_completion": (
        "../leaderboards/hf_space_temporal_condition/"
        "temporal_control_results.json",
        "HumanML3D temporal settings; normalized uTMR FID",
    ),
    "kinematic_motion_control": (
        "../leaderboards/kinematic_motion_control.md",
        "Native-skeleton protocol; rows remain pending",
    ),
    "part_level_motion_control": (
        "https://huggingface.co/spaces/ZeyuLing/"
        "body-part-condition-humanml3d-leaderboard",
        "HumanML3D part-control metrics; normalized uTMR FID",
    ),
    "motion_editing": (
        "https://huggingface.co/spaces/ZeyuLing/"
        "motion-edit-leaderboard",
        "MotionFix semantic preservation and edit-compliance metrics",
    ),
    "motion_repair": (
        "../leaderboards/hf_space_motion_repair/"
        "motion_repair_results.json",
        "BrokenAMASS pair-validated repair metrics; explicit support tracks",
    ),
    "music_to_dance": (
        "../leaderboards/hf_space_music_to_dance/"
        "music_to_dance_results.json",
        "AIST++ music-to-dance and normalized uTMR FID",
    ),
    "dance_to_music": (
        "../leaderboards/hf_space_dance_to_music/"
        "dance_to_music_results.json",
        "AIST++ dance-to-music beat metrics",
    ),
    "monocular_motion_capture": (
        "../leaderboards/hf_space_monocular_capture/"
        "monocular_capture_results.json",
        "3DPW camera/world-space capture metrics",
    ),
}

TASK_PREVIEW_OVERRIDES = {
    ("flowmdm", "text_to_motion"): (
        "assets/model_zoo/flowmdm/"
        "flowmdm_text_to_motion_512_30fps.gif"
    ),
    ("ardy", "text_to_motion"): (
        "assets/model_zoo/ardy/ardy_text_to_motion_512_20fps.gif"
    ),
    ("ardy", "sequential_text_to_motion"): (
        "assets/model_zoo/ardy/ardy_sequential_text_to_motion_512_20fps.gif"
    ),
    ("ardy", "kinematic_motion_control"): (
        "assets/model_zoo/ardy/ardy_kinematic_motion_control_512_30fps.gif"
    ),
    ("condmdi", "temporal_motion_completion"): (
        "assets/model_zoo/condmdi/"
        "condmdi_temporal_motion_completion_512_30fps.gif"
    ),
    ("condmdi", "kinematic_motion_control"): (
        "assets/model_zoo/condmdi/"
        "condmdi_kinematic_motion_control_512_30fps.gif"
    ),
    ("flowmdm", "temporal_motion_completion"): (
        "assets/model_zoo/flowmdm/"
        "flowmdm_temporal_motion_completion_512_30fps.gif"
    ),
    ("flowmdm", "sequential_text_to_motion"): (
        "assets/model_zoo/flowmdm/"
        "flowmdm_sequential_text_to_motion_512_30fps.gif"
    ),
    ("kimodo", "temporal_motion_completion"): (
        "assets/model_zoo/kimodo/"
        "kimodo_temporal_motion_completion_512_30fps.gif"
    ),
    ("kimodo", "sequential_text_to_motion"): (
        "assets/model_zoo/kimodo/"
        "kimodo_sequential_text_to_motion_512_30fps.gif"
    ),
    ("kimodo", "kinematic_motion_control"): (
        "assets/model_zoo/kimodo/"
        "kimodo_kinematic_motion_control_512_30fps.gif"
    ),
    ("maskcontrol", "temporal_motion_completion"): (
        "assets/model_zoo/maskcontrol/"
        "maskcontrol_temporal_motion_completion_512_30fps.gif"
    ),
    ("maskcontrol", "kinematic_motion_control"): (
        "assets/model_zoo/maskcontrol/"
        "maskcontrol_kinematic_motion_control_512_30fps.gif"
    ),
    ("maskcontrol", "part_level_motion_control"): (
        "assets/model_zoo/maskcontrol/"
        "maskcontrol_part_level_motion_control_512_30fps.gif"
    ),
    ("motioncanvas", "temporal_motion_completion"): (
        "assets/model_zoo/motioncanvas/"
        "keyframe_control_512_30fps.gif"
    ),
    ("motioncanvas", "kinematic_motion_control"): (
        "assets/model_zoo/motioncanvas/"
        "trajectory_control_512_30fps.gif"
    ),
    ("motioncanvas", "motion_editing"): (
        "assets/model_zoo/motioncanvas/"
        "instruction_editing_512_30fps.gif"
    ),
    ("motioncanvas", "text_to_motion"): (
        "assets/model_zoo/motioncanvas/"
        "motioncanvas_humanml3d_004822_smpl_mesh_512_30fps.gif"
    ),
    ("motioncanvas", "motion_repair"): (
        "assets/model_zoo/motioncanvas/"
        "motioncanvas_motion_repair_512_20fps.gif"
    ),
    ("humanoid_gpt", "motion_tracking"): (
        "assets/model_zoo/humanoid_gpt/"
        "humanoid_gpt_motion_tracking_512_30fps.gif"
    ),
    ("any2track", "motion_tracking"): (
        "assets/model_zoo/any2track/"
        "any2track_motion_tracking_512_30fps.gif"
    ),
    ("protomotions", "motion_tracking"): (
        "assets/model_zoo/protomotions/"
        "protomotions_motion_tracking_512_30fps.gif"
    ),
    ("sonic", "motion_tracking"): (
        "assets/model_zoo/sonic/"
        "sonic_motion_tracking_512_30fps.gif"
    ),
    ("beyondmimic", "motion_tracking"): (
        "assets/model_zoo/beyondmimic/"
        "beyondmimic_motion_tracking_512_30fps.gif"
    ),
    ("motionstreamer", "temporal_motion_completion"): (
        "assets/model_zoo/motionstreamer/"
        "motionstreamer_temporal_motion_completion_512_30fps.gif"
    ),
    ("motionstreamer", "sequential_text_to_motion"): (
        "assets/model_zoo/motionstreamer/"
        "motionstreamer_sequential_text_to_motion_512_30fps.gif"
    ),
    ("omnicontrol", "text_to_motion"): (
        "assets/model_zoo/omnicontrol/"
        "omnicontrol_text_to_motion_512_30fps.gif"
    ),
    ("omnicontrol", "temporal_motion_completion"): (
        "assets/model_zoo/omnicontrol/"
        "omnicontrol_temporal_motion_completion_512_30fps.gif"
    ),
    ("omnicontrol", "kinematic_motion_control"): (
        "assets/model_zoo/omnicontrol/"
        "omnicontrol_kinematic_motion_control_512_30fps.gif"
    ),
    ("prism", "temporal_motion_completion"): (
        "assets/model_zoo/prism_kt/"
        "prism_temporal_motion_completion_512_30fps.gif"
    ),
    ("prism", "sequential_text_to_motion"): (
        "assets/model_zoo/prism_kt/"
        "prism_sequential_text_to_motion_512_30fps.gif"
    ),
    ("projflow", "temporal_motion_completion"): (
        "assets/model_zoo/projflow/"
        "projflow_temporal_motion_completion_512_20fps.gif"
    ),
    ("projflow", "kinematic_motion_control"): (
        "assets/model_zoo/projflow/"
        "projflow_kinematic_motion_control_512_20fps.gif"
    ),
    ("projflow", "part_level_motion_control"): (
        "assets/model_zoo/projflow/"
        "projflow_part_level_motion_control_512_20fps.gif"
    ),
}

TASK_INPUTS = {
    "motion_to_text": "SMPL motion input",
    "sequential_text_to_motion": (
        "“A person walks forward, then lifts.” → “A person walks back.”"
    ),
    "temporal_motion_completion": (
        "Swaggering walk with observed frames / keyframes"
    ),
    "kinematic_motion_control": "Text plus spatial motion constraints",
    "part_level_motion_control": "Text plus a body-part timeline",
    "music_to_dance": "Break music",
    "dance_to_music": "AIST++ dance motion",
    "monocular_motion_capture": "Monocular RGB video",
    "motion_tracking": "Method-native reference motion and controller state",
}

RUNTIME_ONLY_TASKS = set()

MOTION_TRACKING_PRESENTATION = {
    "any2track": {
        "viewer": (
            "https://zeyuling-motion-tracking-mujoco-leaderboard.static."
            "hf.space/cases/index.html?method=any2track"
        ),
        "leaderboard": (
            "https://huggingface.co/spaces/ZeyuLing/"
            "motion-tracking-mujoco-leaderboard"
        ),
    },
    "protomotions": {
        "viewer": (
            "https://zeyuling-motion-tracking-mujoco-leaderboard.static."
            "hf.space/cases/index.html?method=protomotions"
        ),
        "leaderboard": (
            "https://huggingface.co/spaces/ZeyuLing/"
            "motion-tracking-mujoco-leaderboard"
        ),
    },
    "humanoid_gpt": {
        "viewer": (
            "https://zeyuling-motion-tracking-mujoco-leaderboard.static."
            "hf.space/cases/index.html?method=humanoid_gpt"
        ),
        "leaderboard": (
            "https://huggingface.co/spaces/ZeyuLing/"
            "motion-tracking-mujoco-leaderboard"
        ),
    },
    "sonic": {
        "viewer": (
            "https://zeyuling-motion-tracking-isaaclab-leaderboard.static."
            "hf.space/cases/index.html?method=sonic"
        ),
        "leaderboard": (
            "https://huggingface.co/spaces/ZeyuLing/"
            "motion-tracking-isaaclab-leaderboard"
        ),
    },
    "beyondmimic": {
        "viewer": (
            "https://zeyuling-motion-tracking-isaaclab-leaderboard.static."
            "hf.space/cases/index.html?method=beyondmimic"
        ),
        "leaderboard": (
            "https://huggingface.co/spaces/ZeyuLing/"
            "motion-tracking-isaaclab-leaderboard"
        ),
    },
}

LOCAL_ANIMATION_GLOBS = (
    "*.gif",
    "*.webp",
)

REPOSITORY_MEDIA_PREFIXES = (
    "https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/",
    "https://github.com/ZeyuLing/Motius/blob/main/assets/",
    "https://github.com/ZeyuLing/Motius/raw/main/assets/",
)

CASE_ID_HEADERS = {
    "Sample",
    "HumanML3D Sample",
    "AIST++ Sample",
    "AIST++ Case",
}


def _replace_block(
    text: str,
    start: str,
    end: str,
    replacement: str,
) -> str:
    pattern = re.compile(
        rf"{re.escape(start)}.*?{re.escape(end)}\n?",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(replacement.rstrip() + "\n", text, count=1)
    return text


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _format_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _drop_case_id_columns(text: str) -> str:
    """Remove release-internal dataset IDs from user-facing demo tables."""

    lines = text.splitlines()
    line_number = 0
    while line_number + 1 < len(lines):
        line = lines[line_number]
        if not line.startswith("|"):
            line_number += 1
            continue
        header = _table_cells(line)
        divider = _table_cells(lines[line_number + 1])
        if (
            not header
            or header[0] not in CASE_ID_HEADERS
            or len(divider) != len(header)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in divider)
        ):
            line_number += 1
            continue

        row_number = line_number
        while row_number < len(lines) and lines[row_number].startswith("|"):
            cells = _table_cells(lines[row_number])
            if len(cells) == len(header):
                del cells[0]
                if row_number == line_number and cells[0] in {
                    "Input Text",
                    "Selected Input Text",
                }:
                    cells[0] = "Input"
                lines[row_number] = _format_table_row(cells)
            row_number += 1
        line_number = row_number
    return "\n".join(lines).rstrip() + "\n"


def _repository_relative_media(text: str) -> str:
    for prefix in REPOSITORY_MEDIA_PREFIXES:
        text = text.replace(prefix, "../../assets/")
    return text


def _strip_visible_case_ids(text: str) -> str:
    start = text.find("## Visual Results")
    end = text.find("## Model Overview", start)
    if start < 0 or end < 0:
        return text
    section = text[start:end]
    section = re.sub(r"`M?\d{6}`\s*·\s*", "", section)
    section = re.sub(r"`g[A-Za-z0-9_]+`\s*·\s*", "", section)
    section = re.sub(r"\s*·\s*`m[A-Za-z0-9_]+`", "", section)
    section = re.sub(
        r"(!\[[^\]]*?)\s+HumanML3D\s+M?\d{6}",
        r"\1",
        section,
    )
    return text[:start] + section + text[end:]


def _matching_mp4(image: Path) -> Path | None:
    direct_name = re.sub(
        r"_(?:512|1024)_(?:20|30)fps\.(?:gif|webp)$",
        ".mp4",
        image.name,
    )
    direct = image.with_name(direct_name)
    if direct != image and direct.is_file():
        return direct
    same_stem = image.with_suffix(".mp4")
    if same_stem.is_file():
        return same_stem

    case_id = re.search(r"(M?\d{6})", image.name)
    if case_id is None:
        return None
    candidates = sorted(
        image.parent.glob(f"*{case_id.group(1)}*_smpl_mesh.mp4")
    )
    return candidates[0] if len(candidates) == 1 else None


def _video_attachment_manifest() -> dict[str, dict[str, str]]:
    if not VIDEO_ATTACHMENTS.is_file():
        return {}
    payload = json.loads(VIDEO_ATTACHMENTS.read_text(encoding="utf-8"))
    videos = payload.get("videos", {})
    if not isinstance(videos, dict):
        raise ValueError(f"{VIDEO_ATTACHMENTS}: videos must be an object")
    return videos


def _video_url_for(preview: Path) -> str:
    source = preview.relative_to(REPO_ROOT).as_posix()
    entry = _video_attachment_manifest().get(source)
    if not entry:
        raise FileNotFoundError(
            f"{source}: GitHub video attachment is not published; run "
            "tools/publish_model_card_videos.py"
        )
    url = entry.get("url")
    if not isinstance(url, str) or not re.fullmatch(
        r"https://github\.com/user-attachments/assets/"
        r"[0-9a-fA-F-]{36}",
        url,
    ):
        raise ValueError(f"{source}: invalid GitHub video attachment URL")
    return url


def _video_embed(preview: Path) -> str:
    return f'<video src="{_video_url_for(preview)}" controls></video>'


def _replace_inline_previews_with_video(
    text: str,
    card_path: Path,
) -> str:
    source = r"(\.\./\.\./assets/model_zoo/[^)]+\.(?:gif|webp))"
    nested = re.compile(
        rf"\[!\[[^\]]*\]\({source}\)\]\([^)]+\)",
        flags=re.IGNORECASE,
    )
    plain = re.compile(
        rf"!\[[^\]]*\]\({source}\)",
        flags=re.IGNORECASE,
    )
    html_image = re.compile(
        r'<img\s+[^>]*src="'
        r'(\.\./\.\./assets/model_zoo/[^"]+\.(?:gif|webp))'
        r'"[^>]*>',
        flags=re.IGNORECASE,
    )

    def replacement(match: re.Match[str]) -> str:
        preview = (card_path.parent / match.group(1)).resolve()
        return _video_embed(preview)

    text = nested.sub(replacement, text)
    text = plain.sub(replacement, text)
    return html_image.sub(replacement, text)


def _normalize_preview_media(text: str, card_path: Path) -> str:
    text = _repository_relative_media(text)
    text = _drop_case_id_columns(text)
    text = _strip_visible_case_ids(text)
    text = _replace_inline_previews_with_video(text, card_path)
    return _normalize_t2m_gallery_inputs(text)


def _selected_t2m_captions() -> dict[str, str]:
    manifest = json.loads(
        VIEWER_MANIFESTS["text_to_motion"].read_text(encoding="utf-8")
    )
    captions = {}
    for case in manifest.get("cases", []):
        references = case.get("references") or []
        if not references or not isinstance(references[0], str):
            continue
        for key in ("case_id", "sample_id", "case_key"):
            value = case.get(key)
            if value is not None:
                captions[str(value)] = references[0].strip()
    return captions


def _attachment_sources_by_url() -> dict[str, str]:
    if not VIDEO_ATTACHMENTS.is_file():
        return {}
    payload = json.loads(VIDEO_ATTACHMENTS.read_text(encoding="utf-8"))
    return {
        entry["url"]: source
        for source, entry in payload.get("videos", {}).items()
        if isinstance(entry, dict) and isinstance(entry.get("url"), str)
    }


def _normalize_t2m_gallery_inputs(text: str) -> str:
    captions = _selected_t2m_captions()
    sources = _attachment_sources_by_url()
    row = re.compile(
        r"^\| (?P<input>[^|\n]+) \| "
        r'(?P<video><video src="(?P<url>[^"]+)" controls></video>) \|$',
        flags=re.MULTILINE,
    )
    case_pattern = re.compile(
        r"_humanml3d_(?P<case>[^/]+?)_smpl_mesh"
        r"(?:_(?:512|1024)_(?:20|30)fps)?\.(?:gif|webp)$"
    )

    def replacement(match: re.Match[str]) -> str:
        source = sources.get(match.group("url"))
        case_match = case_pattern.search(source or "")
        if case_match is None:
            return match.group(0)
        caption = captions.get(case_match.group("case"))
        if not caption:
            return match.group(0)
        caption = caption.replace("|", "\\|")
        return f"| {caption} | {match.group('video')} |"

    return row.sub(replacement, text)


def _viewer_methods(task: str) -> set[str]:
    path = VIEWER_MANIFESTS.get(task)
    if path is None:
        return set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    methods = manifest.get("motion_methods", [])
    if task == "motion_to_text":
        methods = manifest.get("output_methods", [])
    return {entry["key"] for entry in methods}


def _temporal_methods() -> set[str]:
    root = (
        REPO_ROOT
        / "docs/leaderboards/hf_space_temporal_condition/cases"
    )
    methods: set[str] = set()
    for path in root.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        methods.update(
            entry["key"] for entry in manifest.get("motion_methods", [])
        )
    return methods


def _animation_directory(package: str) -> Path:
    directory = REPO_ROOT / "assets" / "model_zoo" / package
    if not directory.is_dir() and package == "prism":
        directory = REPO_ROOT / "assets" / "model_zoo" / "prism_kt"
    return directory


def _first_local_animation(package: str, task: str) -> Path | None:
    directory = _animation_directory(package)
    if not directory.is_dir():
        return None
    candidates = [
        path
        for pattern in LOCAL_ANIMATION_GLOBS
        for path in sorted(directory.glob(pattern))
    ]
    if task == "text_to_motion":
        preferred = [
            path for path in candidates if "humanml3d" in path.name.lower()
        ]
        candidates = preferred or candidates
    elif task == "music_to_dance":
        preferred = [
            path for path in candidates if "aistpp" in path.name.lower()
        ]
        candidates = preferred or candidates
    candidates.sort(
        key=lambda path: (
            "_512_" not in path.name,
            path.suffix.lower() != ".gif",
            "roundhouse_kick" in path.name,
            path.name,
        )
    )
    return candidates[0] if candidates else None


def _preview_for(package: str, task: str) -> Path:
    override = TASK_PREVIEW_OVERRIDES.get((package, task))
    if override is not None:
        preview = REPO_ROOT / override
    elif task == "motion_to_text":
        preview = (
            REPO_ROOT
            / "assets/model_zoo/shared/"
            "motion_to_text_input_smpl_512_30fps.gif"
        )
    elif task == "dance_to_music" and package == "unimumo":
        preview = (
            REPO_ROOT
            / "assets/model_zoo/unimumo/"
            "unimumo_dance_to_music_input_smpl_512_30fps.gif"
        )
    else:
        preview = _first_local_animation(package, task)
    if preview is None or not preview.is_file():
        target = override or f"assets/model_zoo/{package}/<animation>"
        raise FileNotFoundError(
            f"{package}/{task}: rendered task preview is missing: {target}"
        )
    return preview


def _viewer_for(package: str, task: str) -> str | None:
    key = METHOD_KEYS.get(package, package)
    if task == "temporal_motion_completion" and key in _temporal_methods():
        return VIEWER_BASE[task] + f"?method={key}"
    if task == "temporal_motion_completion" and package in {
        "motionstreamer",
        "prism",
    }:
        return (
            VIEWER_BASE["sequential_text_to_motion"]
            + f"?method={package}"
        )
    if task == "part_level_motion_control" and package == "maskcontrol":
        return (
            "https://huggingface.co/spaces/ZeyuLing/"
            "body-part-condition-humanml3d-leaderboard"
        )

    methods = _viewer_methods(task)
    if key not in methods and package in methods:
        key = package
    if key in methods:
        return VIEWER_BASE[task] + f"?method={key}"

    if task == "dance_to_music" and package == "unimumo":
        return VIEWER_BASE[task]
    return None


def _preview_metadata(preview: Path) -> dict:
    stem = re.sub(r"_(?:512|1024)_(?:20|30)fps$", "", preview.stem)
    candidates = [
        preview.with_suffix(".json"),
        preview.with_name(f"{stem}.json"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _input_for(package: str, task: str, preview: Path) -> str:
    metadata = _preview_metadata(preview)
    task_input = metadata.get("input") or metadata.get("caption")
    condition = metadata.get("condition")
    if isinstance(task_input, str) and task_input.strip():
        rendered = task_input.strip()
        if isinstance(condition, str) and condition.strip():
            rendered += f"<br><sub>{condition.strip()}</sub>"
        return rendered
    if package == "ardy" and task == "text_to_motion":
        return "A person walks forward, turns right, and starts jogging."
    return TASK_INPUTS.get(task, "Native task input")


def _relative_media(path: Path) -> str:
    return (Path("../..") / path.relative_to(REPO_ROOT)).as_posix()


def _more_links(package: str, task: str, preview: Path) -> str:
    links = [f"[MP4]({_video_url_for(preview)})"]
    viewer = _preview_metadata(preview).get("viewer_url")
    if not isinstance(viewer, str) or not viewer.strip():
        viewer = _viewer_for(package, task)
    if viewer is not None:
        links.append(f"[All cases]({viewer})")
    return " · ".join(links) if links else "—"


def _task_demo_block(
    package: str,
    tasks: list[str],
) -> str:
    labels = {
        task["id"]: task["label"] for task in TASK_REGISTRY["tasks"]
    }
    rows = [
        DEMO_START,
        "",
        "### Task Demos",
        "",
        "| Task | Input / condition | Rendered output | More |",
        "| --- | --- | --- | --- |",
    ]
    if not tasks:
        preview = _first_local_animation(package, "")
        if preview is None:
            rows.append(
                "| No registered public task | See the capability boundary "
                "below | No redistributable repository preview | — |"
            )
        else:
            media = _video_embed(preview)
            rows.append(
                "| No registered public task | Native capability input | "
                f"{media} | — |"
            )
    for task in tasks:
        if task == "motion_tracking":
            presentation = MOTION_TRACKING_PRESENTATION[package]
            label = labels[task]
            preview = _preview_for(package, task)
            rendered = _video_embed(preview)
            more = (
                f"[MP4]({_video_url_for(preview)}) · "
                f"[All cases]({presentation['viewer']}) · "
                f"[Leaderboard]({presentation['leaderboard']})"
            )
            rows.append(
                f"| {label} | {TASK_INPUTS[task]} | {rendered} | {more} |"
            )
            continue
        if task in RUNTIME_ONLY_TASKS:
            rows.append(
                f"| {labels[task]} | {TASK_INPUTS[task]} | "
                "Physical rollout preview pending a shared simulator protocol | "
                "— |"
            )
            continue
        preview = _preview_for(package, task)
        label = labels[task]
        media = _video_embed(preview)
        task_input = _input_for(package, task, preview).replace("|", "\\|")
        rows.append(
            f"| {label} | {task_input} | {media} | "
            f"{_more_links(package, task, preview)} |"
        )
    if tasks == ["motion_tracking"]:
        note = (
            "The policy-step API and physical rollout are evaluated under the "
            "registered MuJoCo or Isaac Lab protocol stated in the Model Card."
        )
    else:
        note = (
            "Every public `infer_*` API is represented by a GitHub-native "
            "H.264 video player. **All cases** opens the optional interactive "
            "comparison."
        )
    rows.extend(["", note, "", DEMO_END])
    return "\n".join(rows)


def _frame_rate_contract_block(package: str) -> str:
    contract = FRAME_RATE_CONTRACTS[package]
    return "\n".join(
        [
            FRAME_RATE_START,
            "",
            "### Frame-Rate Contract",
            "",
            "| Clock | Rate |",
            "| --- | --- |",
            f"| Training motion | {contract['training']} |",
            f"| Public preview | {contract['preview']} |",
            "",
            (
                "Training FPS is the checkpoint's native temporal clock. "
                "Preview FPS only controls media playback; any conversion "
                "listed above preserves duration."
            ),
            "",
            FRAME_RATE_END,
        ]
    )


def _semantic_index() -> dict[tuple[str, str], dict]:
    payload = json.loads(T2M_RESULTS.read_text(encoding="utf-8"))
    return {
        (row["method"], row["version"]): row
        for row in payload["semantic_rows"]
    }


def _row_for_variant(
    package: str,
    variant: str,
    index: dict[tuple[str, str], dict],
) -> dict | None:
    variants = T2M_ROW_KEYS.get(package)
    if not variants:
        return None
    key = variants.get(variant.lower()) or variants.get("default")
    return index.get(key) if key else None


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _model_zoo_metrics_block(
    index: dict[tuple[str, str], dict],
) -> str:
    package_by_t2m_key = {
        key: package
        for package, variants in T2M_ROW_KEYS.items()
        for key in variants.values()
    }
    display_overrides = {
        "hymotion_t2m": "HY-Motion T2M",
        "motionmillion": "MotionMillion / GoToZero",
    }
    t2m_payload = json.loads(T2M_RESULTS.read_text(encoding="utf-8"))
    t2m_rows = []
    for row in t2m_payload["semantic_rows"]:
        key = (row["method"], row["version"])
        package = package_by_t2m_key.get(key)
        if package is None or row.get("isReference", False):
            continue
        method = display_overrides.get(package, row["method"])
        values = [
            f"[{method}]({package}.md)",
            row["version"],
            _number(row.get("msR3")),
            _number(row.get("msFID")),
            _number(row.get("utmrR3")),
            _number(row.get("utmrFIDNorm")),
        ]
        t2m_rows.append(_format_table_row(values))

    m2d_packages = {
        "bailando": "bailando",
        "edge": "edge",
        "tm2d": "tm2d",
        "unimumo": "unimumo",
    }
    m2d_payload = json.loads(M2D_RESULTS.read_text(encoding="utf-8"))
    m2d_rows = []
    for row in m2d_payload["rows"]:
        package = m2d_packages.get(row["method"].lower())
        if package is None or row.get("reference", False):
            continue
        values = [
            f"[{row['method']}]({package}.md)",
            row["version"],
            _number(row.get("fid_k")),
            _number(row.get("fid_g")),
            _number(row.get("fid_utmr")),
            _number(row.get("beat_align")),
        ]
        m2d_rows.append(_format_table_row(values))

    rows = [
        ZOO_METRICS_START,
        "",
        "## Canonical Benchmark Snapshot",
        "",
        (
            "These compact rows are generated from the same machine-readable "
            "snapshots as the public Leaderboards and Model Cards. "
            "`Motius FID` is always computed in per-sample L2-normalized "
            "embedding space; `—` means that value has not been recomputed."
        ),
        "",
        "### Text-to-Motion · SMPL Skeleton",
        "",
        "| Method | Version | MotionStreamer R@3 ↑ | MotionStreamer FID ↓ | "
        "Motius R@3 ↑ | Motius FID (normalized) ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *t2m_rows,
        "",
        (
            "[Open the complete Text-to-Motion Leaderboard]"
            "(https://huggingface.co/spaces/ZeyuLing/"
            "t2m-humanml3d-leaderboard) for all three evaluators, physical "
            "metrics, protocol details, and case-level SMPL visualization."
        ),
        "",
        "### Music-to-Dance · AIST++",
        "",
        "| Method | Version | FID_k ↓ | FID_g ↓ | "
        "Motius FID (normalized) ↓ | BeatAlign ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *m2d_rows,
        "",
        (
            "[Open the complete Music-to-Dance Leaderboard]"
            "(https://huggingface.co/spaces/ZeyuLing/"
            "music-to-dance-aistpp-leaderboard) for diversity, physical "
            "metrics, synchronized audio, and interactive 3D results."
        ),
        "",
        ZOO_METRICS_END,
    ]
    return "\n".join(rows)


def _m2t_snapshot(package: str) -> list[str]:
    payload = json.loads(M2T_RESULTS.read_text(encoding="utf-8"))
    row = next(
        (
            item
            for item in payload["methods"]
            if item["id"] == package and item.get("kind") == "method"
        ),
        None,
    )
    if row is None:
        return []
    metrics = row["metrics"]
    values = [
        row["name"],
        f"{payload['benchmark']['num_samples']:,}",
        _number(metrics["bleu1"]),
        _number(metrics["bleu4"]),
        _number(metrics["rougeL"]),
        _number(metrics["cider"]),
        _number(metrics["bertRaw"]),
        _number(metrics["bertF1"]),
        _number(metrics["r1"]),
        _number(metrics["r2"]),
        _number(metrics["r3"]),
        _number(metrics["matching"]),
    ]
    return [
        "### Canonical Motion-to-Text Snapshot",
        "",
        "| Method | n | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | "
        "BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
        "| " + " | ".join(values) + " |",
    ]


def _m2d_snapshot(package: str) -> list[str]:
    payload = json.loads(M2D_RESULTS.read_text(encoding="utf-8"))
    row = next(
        (
            item
            for item in payload["rows"]
            if item["method"].lower() == package
            and not item.get("reference", False)
        ),
        None,
    )
    if row is None:
        return []
    values = [
        row["method"],
        f"{payload['population']:,}",
        _number(row["fid_k"]),
        _number(row["fid_g"]),
        _number(row["fid_utmr"]),
        _number(row["diversity_k"]),
        _number(row["diversity_g"]),
        _number(row["beat_align"]),
    ]
    return [
        "### Canonical Music-to-Dance Snapshot",
        "",
        "| Method | n | FID_k | FID_g | Motius FID (normalized) | "
        "Diversity_k | Diversity_g | BeatAlign |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| " + " | ".join(values) + " |",
    ]


def _d2m_snapshot(package: str) -> list[str]:
    payload = json.loads(D2M_RESULTS.read_text(encoding="utf-8"))
    row = next(
        (
            item
            for item in payload["rows"]
            if item["method"].lower() == package
            and not item.get("reference", False)
        ),
        None,
    )
    if row is None:
        return []
    values = [
        row["method"],
        f"{payload['population']:,}",
        _number(row["beat_count_ratio"]),
        _number(row["beat_hit_rate"]),
    ]
    return [
        "### Canonical Dance-to-Music Snapshot",
        "",
        "| Method | n | Beat count ratio | Beat hit rate |",
        "| --- | ---: | ---: | ---: |",
        "| " + " | ".join(values) + " |",
    ]


def _temporal_snapshot(package: str) -> list[str]:
    payload = json.loads(TEMPORAL_RESULTS.read_text(encoding="utf-8"))
    method_id = TEMPORAL_METHOD_IDS.get(package, package)
    control_rows = []
    for setting in payload["settings"]:
        method = next(
            (
                item
                for item in setting["methods"]
                if item["method_id"] == method_id
            ),
            None,
        )
        if method is None:
            continue
        metrics = method["metrics"]
        values = [
            setting["id"],
            f"{method['samples']:,}",
            _number(metrics["r_precision_top1"]),
            _number(metrics["r_precision_top2"]),
            _number(metrics["r_precision_top3"]),
            _number(metrics["fid"]),
            _number(metrics["mm_dist"]),
            _number(metrics["diversity"]),
            _number(metrics["constraint_error_cm"]),
            _number(metrics["foot_skating"]),
        ]
        control_rows.append("| " + " | ".join(values) + " |")

    tp2m_name = TP2M_METHODS.get(package)
    tp2m_rows = [
        row
        for row in payload.get("tp2m_rows", [])
        if row["method"] == tp2m_name
    ]
    if not control_rows and not tp2m_rows:
        return []
    rows = [
        "### Canonical Temporal-Completion Snapshot",
    ]
    if control_rows:
        rows.extend(
            [
                "",
                "#### Temporal Control · Motius normalized space",
                "",
                "| Setting | n | R@1 | R@2 | R@3 | "
                "Motius FID (normalized) | MM-Dist | Diversity | "
                "Constraint error (cm) | Foot skating |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |",
                *control_rows,
            ]
        )
    if tp2m_rows:
        rows.extend(
            [
                "",
                "#### TP2M Prefix · MotionStreamer-272 space",
                "",
                "| Setting | n | R@1 | R@2 | R@3 | MotionStreamer FID | "
                "MM-Dist | Diversity |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: |",
            ]
        )
        for row in tp2m_rows:
            values = [
                row["settingLabel"],
                f"{row['samples']:,}",
                _number(row["r1"]),
                _number(row["r2"]),
                _number(row["r3"]),
                _number(row["fid"]),
                _number(row["mmDist"]),
                _number(row["diversity"]),
            ]
            rows.append("| " + " | ".join(values) + " |")
    return rows


def _monocular_snapshot(package: str) -> list[str]:
    method_name = MONOCULAR_METHODS.get(package)
    if method_name is None:
        return []
    payload = json.loads(MONOCULAR_RESULTS.read_text(encoding="utf-8"))
    method = next(
        (
            item
            for item in payload["methods"]
            if item["method"] == method_name
        ),
        None,
    )
    if method is None:
        return []
    rows = []
    for protocol, result in method.get("verified_results", {}).items():
        metrics = result["metrics"]
        rows.append(
            "| "
            + " | ".join(
                [
                    protocol,
                    _number(result.get("coverage_percent")),
                    _number(metrics.get("pa_mpjpe_mm")),
                    _number(metrics.get("mpjpe_mm")),
                    _number(metrics.get("pve_mm")),
                    _number(metrics.get("accel_mps2")),
                ]
            )
            + " |"
        )
    if not rows:
        return []
    return [
        "### Canonical Monocular-Capture Snapshot",
        "",
        "| Protocol | Coverage (%) | PA-MPJPE (mm) | MPJPE (mm) | "
        "PVE (mm) | Accel (m/s²) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
    ]


def _motion_repair_snapshot(package: str) -> list[str]:
    if package != "motioncanvas":
        return []
    payload = json.loads(MOTION_REPAIR_RESULTS.read_text(encoding="utf-8"))
    result = next(
        (
            row
            for row in payload["rows"]
            if row["method"] == "MotionCanvas"
        ),
        None,
    )
    if result is None:
        return []
    return [
        "### Canonical Motion-Repair Snapshot",
        "",
        "| Support | n | uTMR R@1 | uTMR R@3 | uTMR M2M | "
        "MPJPE (cm) | Accel. error | Jitter |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                result["support"],
                f"{result['samples']:,}",
                _number(result["utmr_r1"]),
                _number(result["utmr_r3"]),
                _number(result["utmr_m2m"]),
                _number(result["mpjpe_cm"]),
                _number(result["accel_error"]),
                _number(result["jitter"]),
            ]
        )
        + " |",
    ]


def _task_metric_snapshots(package: str, tasks: list[str]) -> list[str]:
    builders = {
        "motion_to_text": _m2t_snapshot,
        "temporal_motion_completion": _temporal_snapshot,
        "motion_repair": _motion_repair_snapshot,
        "music_to_dance": _m2d_snapshot,
        "dance_to_music": _d2m_snapshot,
        "monocular_motion_capture": _monocular_snapshot,
    }
    rows: list[str] = []
    for task in tasks:
        builder = builders.get(task)
        snapshot = builder(package) if builder else []
        if snapshot:
            rows.extend(["", *snapshot])
    return rows


def _canonical_row(
    evaluator: str,
    variant: str,
    row: dict,
    with_variant: bool,
    with_status: bool,
) -> str:
    if evaluator == "MotionStreamer Evaluator":
        fields = [
            f"{row['msN']:,}" if row.get("msN") is not None else "—",
            _number(row.get("msR1")),
            _number(row.get("msR2")),
            _number(row.get("msR3")),
            _number(row.get("msFID")),
            _number(row.get("msMM")),
            _number(row.get("msDiv")),
        ]
        status = "Measured" if row.get("msN") is not None else "Not measured"
    else:
        fields = [
            f"{row['utmrN']:,}" if row.get("utmrN") is not None else "—",
            _number(row.get("utmrR1")),
            _number(row.get("utmrR2")),
            _number(row.get("utmrR3")),
            _number(row.get("utmrFIDNorm")),
            _number(row.get("utmrMM")),
            _number(row.get("utmrDiv")),
        ]
        status = (
            "Measured"
            if row.get("utmrFIDNorm") is not None
            else "Normalized FID not recomputed"
        )
    cells = [evaluator]
    if with_variant:
        cells.append(variant)
    cells.extend(fields)
    if with_status:
        cells.append(status)
    return "| " + " | ".join(cells) + " |"


def _sync_existing_t2m_rows(
    text: str,
    package: str,
    index: dict[tuple[str, str], dict],
) -> tuple[str, bool]:
    lines = text.splitlines()
    changed = False
    found = False
    inside_generated_block = False
    for line_number, line in enumerate(lines):
        if line == METRICS_START:
            inside_generated_block = True
            continue
        if line == METRICS_END:
            inside_generated_block = False
            continue
        if inside_generated_block:
            continue
        if not line.startswith(
            ("| MotionStreamer Evaluator |", "| Motius Joint-Position Evaluator |")
        ):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        evaluator = cells[0]
        with_variant = len(cells) >= 9
        with_status = len(cells) == 10
        variant = cells[1] if with_variant else "Default"
        row = _row_for_variant(package, variant, index)
        if row is None:
            continue
        replacement = _canonical_row(
            evaluator,
            variant,
            row,
            with_variant,
            with_status,
        )
        found = True
        if replacement != line:
            lines[line_number] = replacement
            changed = True
    return "\n".join(lines).rstrip() + "\n", found


def _motion_tracking_metrics_block(package: str) -> str:
    specs = {
        "any2track": (MOTION_TRACKING_MUJOCO_RESULTS, "Any2Track"),
        "protomotions": (MOTION_TRACKING_MUJOCO_RESULTS, "ProtoMotions"),
        "humanoid_gpt": (MOTION_TRACKING_MUJOCO_RESULTS, "HumanoidGPT"),
        "sonic": (MOTION_TRACKING_ISAACLAB_RESULTS, "SONIC"),
        "beyondmimic": (MOTION_TRACKING_ISAACLAB_RESULTS, "BeyondMimic"),
    }
    path, method = specs[package]
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(item for item in payload["rows"] if item["method"] == method)
    split_labels = {
        item["id"]: item["label"] for item in payload["splits"]
    }
    engine = payload["engine"]["name"]
    rows = [
        METRICS_START,
        "",
        (
            f"> **Measured physical rollout.** Scores use the registered "
            f"{engine} protocol `{payload['protocol_id']}` at "
            f"{payload['control_hz']} Hz."
        ),
        "",
        "| Setting | Coverage | Success ↑ | Completion ↑ | "
        "Local MPJPE ↓ | Joint MAE ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, result in row["splits"].items():
        coverage = result["coverage"]
        metrics = result["metrics"]
        rows.append(
            f"| {engine} · {split_labels[split]} | "
            f"{coverage['evaluated']} / {coverage['population']} | "
            f"{100 * metrics['success_rate']:.1f}% | "
            f"{100 * metrics['completion_rate']:.1f}% | "
            f"{metrics['local_mpjpe_mm']:.2f} mm | "
            f"{metrics['joint_mae_rad']:.4f} rad |"
        )
    if row.get("note"):
        rows.extend(["", f"> {row['note']}"])
    result_link = (
        Path("..") / path.relative_to(REPO_ROOT / "docs")
    ).as_posix()
    rows.extend(
        [
            "",
            f"[Canonical result pack]({result_link})",
            "",
            METRICS_END,
        ]
    )
    return "\n".join(rows)


def _canonical_metrics_block(
    package: str,
    tasks: list[str],
    index: dict[tuple[str, str], dict],
    has_existing_rows: bool,
) -> str:
    if tasks == ["motion_tracking"]:
        return _motion_tracking_metrics_block(package)
    else:
        metrics_note = (
            "> **Canonical metrics.** Public results are tied to the sources "
            "below. Motius/uTMR FID always means per-sample L2-normalized "
            "embedding-space FID; `—` means the normalized value has not been "
            "recomputed. Historical raw-space FID is never substituted."
        )
    rows = [
        METRICS_START,
        "",
        metrics_note,
    ]
    labels = {
        task["id"]: task["label"] for task in TASK_REGISTRY["tasks"]
    }
    rows.extend(
        [
            "",
            "| Task | Canonical result source | Protocol |",
            "| --- | --- | --- |",
        ]
    )
    if not tasks:
        rows.append(
            "| No registered public task | Not applicable | "
            "See the capability boundary |"
        )
    for task in tasks:
        source, protocol = METRIC_SOURCES[task]
        rows.append(
            f"| {labels[task]} | [Published results]({source}) | "
            f"{protocol} |"
        )
    if package == "hymotion_t2m" and "text_to_motion" in tasks:
        rows.append(
            "| Text-to-Motion · Unitree G1 | "
            "[Published results](../leaderboards/hf_space_t2m_unitree_g1/"
            "g1_results.json) | Fixed 1,024-case G1 protocol with TMR-G1 |"
        )
    rows.extend(_task_metric_snapshots(package, tasks))
    if "text_to_motion" in tasks and not has_existing_rows:
        variants = T2M_ROW_KEYS.get(package, {})
        semantic_rows = []
        for variant, key in variants.items():
            if variant == "default":
                variant = key[1]
            row = index.get(key)
            if row is not None:
                semantic_rows.append((variant, row))
        rows.extend(
            [
                "",
                "### Canonical HumanML3D Semantic Results",
                "",
                "| Evaluator | Variant | n | R@1 | R@2 | R@3 | FID | "
                "MM-Dist | Diversity | Status |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | --- |",
            ]
        )
        if not semantic_rows:
            rows.extend(
                [
                    "| HumanML3D Official | Default | — | — | — | — | — | "
                    "— | — | Not measured |",
                    "| MotionStreamer Evaluator | Default | — | — | — | — | "
                    "— | — | — | Not measured |",
                    "| Motius Joint-Position Evaluator | Default | — | — | — | "
                    "— | — | — | — | Not measured |",
                ]
            )
        for variant, row in semantic_rows:
            rows.append(
                "| HumanML3D Official | "
                f"{variant} | — | — | — | — | — | — | — | Not measured |"
            )
            rows.append(
                _canonical_row(
                    "MotionStreamer Evaluator",
                    variant,
                    row,
                    with_variant=True,
                    with_status=True,
                )
            )
            rows.append(
                _canonical_row(
                    "Motius Joint-Position Evaluator",
                    variant,
                    row,
                    with_variant=True,
                    with_status=True,
                )
            )
    rows.extend(["", METRICS_END])
    return "\n".join(rows)


def _insert_after_heading(text: str, heading: str, block: str) -> str:
    marker = heading + "\n"
    if marker not in text:
        raise ValueError(f"Missing heading {heading!r}")
    return text.replace(marker, marker + "\n" + block + "\n\n", 1)


def _insert_before_heading(text: str, heading: str, block: str) -> str:
    marker = heading + "\n"
    if marker not in text:
        raise ValueError(f"Missing heading {heading!r}")
    return text.replace(marker, block + "\n\n" + marker, 1)


def sync_card(
    package: str,
    path: Path,
    tasks: list[str],
    index: dict[tuple[str, str], dict],
) -> str:
    text = path.read_text(encoding="utf-8")
    text = _normalize_preview_media(text, path)
    demos = _task_demo_block(package, tasks)
    text = _replace_block(text, DEMO_START, DEMO_END, demos)
    if DEMO_START not in text:
        text = _insert_after_heading(text, "## Visual Results", demos)

    frame_rate = _frame_rate_contract_block(package)
    text = _replace_block(
        text,
        FRAME_RATE_START,
        FRAME_RATE_END,
        frame_rate,
    )
    if FRAME_RATE_START not in text:
        if TASKS_END not in text:
            raise ValueError(f"{path}: missing task contract marker")
        text = text.replace(
            TASKS_END,
            TASKS_END + "\n\n" + frame_rate,
            1,
        )

    text, has_existing_rows = _sync_existing_t2m_rows(text, package, index)
    metrics = _canonical_metrics_block(
        package,
        tasks,
        index,
        has_existing_rows,
    )
    text = _replace_block(text, METRICS_START, METRICS_END, metrics)
    if METRICS_START not in text:
        text = _insert_after_heading(text, "## Evaluation Results", metrics)
    return text


def sync_all(write: bool) -> list[Path]:
    cards = _catalog_cards()
    contracts = _catalog_task_contracts()
    index = _semantic_index()
    changed = []
    for package, path in sorted(cards.items()):
        rendered = sync_card(package, path, contracts[package], index)
        if rendered != path.read_text(encoding="utf-8"):
            changed.append(path)
            if write:
                path.write_text(rendered, encoding="utf-8")

    zoo_text = MODEL_ZOO_README.read_text(encoding="utf-8")
    zoo_metrics = _model_zoo_metrics_block(index)
    rendered_zoo = _replace_block(
        zoo_text,
        ZOO_METRICS_START,
        ZOO_METRICS_END,
        zoo_metrics,
    )
    if ZOO_METRICS_START not in rendered_zoo:
        rendered_zoo = _insert_before_heading(
            rendered_zoo,
            "## Model Card Standard",
            zoo_metrics,
        )
    if rendered_zoo != zoo_text:
        changed.append(MODEL_ZOO_README)
        if write:
            MODEL_ZOO_README.write_text(rendered_zoo, encoding="utf-8")

    manifest = _sync_release_manifest(index)
    rendered_manifest = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    if rendered_manifest != RELEASE_MANIFEST.read_text(encoding="utf-8"):
        changed.append(RELEASE_MANIFEST)
        if write:
            RELEASE_MANIFEST.write_text(rendered_manifest, encoding="utf-8")
    return changed


def _sync_release_manifest(
    index: dict[tuple[str, str], dict],
) -> dict:
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest["metric_policy"] = {
        "canonical_t2m_source": (
            "../leaderboards/hf_space_t2m_humanml3d/t2m_results.json"
        ),
        "motius_fid_field": "utmrFIDNorm",
        "motius_fid_space": "per-sample L2-normalized embedding space",
        "raw_fid_policy": "not published in Model Cards",
    }
    for model_key, model in manifest.get("models", {}).items():
        package, forced_variant = RELEASE_PACKAGE_ALIASES.get(
            model_key,
            (model_key, None),
        )
        for metric in model.get("metrics", []):
            evaluator = metric.get("evaluator")
            if evaluator not in {
                "MotionStreamer Evaluator",
                "Motius Joint-Position Evaluator",
            }:
                continue
            variant = forced_variant or metric.get("variant", "Default")
            row = _row_for_variant(package, variant, index)
            if row is None:
                continue
            if evaluator == "MotionStreamer Evaluator":
                values = {
                    "samples": row.get("msN"),
                    "R1": row.get("msR1"),
                    "R2": row.get("msR2"),
                    "R3": row.get("msR3"),
                    "FID": row.get("msFID"),
                    "MM-Dist": row.get("msMM"),
                    "Diversity": row.get("msDiv"),
                    "FID_space": "MotionStreamer evaluator embedding space",
                    "status": (
                        "measured"
                        if row.get("msN") is not None
                        else "not_measured"
                    ),
                }
            else:
                values = {
                    "samples": row.get("utmrN"),
                    "R1": row.get("utmrR1"),
                    "R2": row.get("utmrR2"),
                    "R3": row.get("utmrR3"),
                    "FID": row.get("utmrFIDNorm"),
                    "MM-Dist": row.get("utmrMM"),
                    "Diversity": row.get("utmrDiv"),
                    "FID_space": (
                        "per-sample L2-normalized embedding space"
                    ),
                    "status": (
                        "measured"
                        if row.get("utmrFIDNorm") is not None
                        else "normalized_fid_not_recomputed"
                    ),
                }
            metric.update(values)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed = sync_all(write=args.write)
    for path in changed:
        print(path.relative_to(REPO_ROOT))
    if changed and not args.write:
        print(f"{len(changed)} Model Card(s) require content synchronization")
        return 1
    print(
        f"{len(changed)} Model Zoo file(s) synchronized"
        if args.write
        else "All Model Zoo content is synchronized"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
