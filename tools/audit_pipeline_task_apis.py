#!/usr/bin/env python3
"""Audit canonical Motius task APIs and their pre-contract inference routes."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hf_checkpoint_specs import CHECKPOINT_SPECS


# Public method Pipelines that intentionally cannot be distributed as Motius
# Hugging Face artifacts are still part of the task-API contract.
NON_HF_PIPELINE_TASKS = {
    "motius.pipelines.beyondmimic.BeyondMimicPipeline": (
        "motion_tracking",
    ),
    "motius.pipelines.motionrepair.MotionRepairPipeline": (
        "motion_repair",
    ),
    "motius.pipelines.prompthmr.PromptHMRPipeline": (
        "monocular_motion_capture",
    ),
}

# Every Pipeline class in motius/pipelines must either implement a public
# method-task contract or have a precise non-method reason for exclusion.
PIPELINE_CLASS_EXCLUSIONS = {
    "BasePipeline": "abstract pipeline base class",
    "MotionCLIPPipeline": "evaluator/retrieval API, not a method Pipeline",
    "Pipeline": "automatic artifact loader that returns a method Pipeline",
    "PrismARPipeline": "internal Diffusers backend used by PRISMPipeline",
}


# These are the entrypoints used before the canonical infer_{task} contract.
# Tasks absent from this map were already implemented under their canonical
# name and are classified as native.
LEGACY_TASK_ROUTES = {
    "motius.pipelines.ardy.ARDYPipeline": {
        "text_to_motion": "infer_t2m",
        "sequential_text_to_motion": "stream_step",
        "kinematic_motion_control": "generate",
    },
    "motius.pipelines.condmdi.CondMDIPipeline": {
        "text_to_motion": "infer_t2m",
        "kinematic_motion_control": "infer_control",
    },
    "motius.pipelines.dart.DARTPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.flowmdm.FlowMDMPipeline": {
        "text_to_motion": "infer_t2m",
        "temporal_motion_completion": "infer_tp2m",
        "sequential_text_to_motion": "infer_sequential_t2m",
    },
    "motius.pipelines.hymotion_t2m.HyMotionT2MPipeline": {
        "text_to_motion": "__call__",
    },
    "motius.pipelines.intergen.InterGenPipeline": {
        "text_to_multi_person_motion": "infer_t2m",
    },
    "motius.pipelines.intermask.InterMaskPipeline": {
        "text_to_multi_person_motion": "infer_t2m",
    },
    "motius.pipelines.kimodo.KIMODOPipeline": {
        "text_to_motion": "text_to_motion",
        "temporal_motion_completion": "infer_tp2m",
        "sequential_text_to_motion": "multi_prompt",
        "kinematic_motion_control": "constrained_motion",
    },
    "motius.pipelines.maskcontrol.MaskControlPipeline": {
        "text_to_motion": "infer_t2m",
        "temporal_motion_completion": "infer_temporal",
        "sequential_text_to_motion": "infer_sequential",
        "part_level_motion_control": "infer_body_part",
    },
    "motius.pipelines.mdm.MDMPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.mld.MLDPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.mogents.MoGenTSPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.momask.MoMaskPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.motionbricks.MotionBricksPipeline": {
        "g1_realtime_navigation": "rollout",
        "g1_qpos_generation": "rollout",
    },
    "motius.pipelines.motionclr.MotionCLRPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.motiongpt.MotionGPTPipeline": {
        "text_to_motion": "infer_t2m",
        "motion_to_text": "infer_m2t",
    },
    "motius.pipelines.motiongpt3.MotionGPT3Pipeline": {
        "text_to_motion": "infer_t2m",
        "motion_to_text": "infer_m2t",
    },
    "motius.pipelines.motionlcm.MotionLCMPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.motionmillion.MotionMillionPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.motionstreamer.MotionStreamerPipeline": {
        "text_to_motion": "infer_t2m",
        "temporal_motion_completion": "infer_tp2m",
        "sequential_text_to_motion": "infer_sequential_t2m",
    },
    "motius.pipelines.omnicontrol.OmniControlPipeline": {
        "text_to_motion": "infer_t2m",
        "temporal_motion_completion": "infer_control",
        "kinematic_motion_control": "infer_control",
    },
    "motius.pipelines.prism.PRISMPipeline": {
        "text_to_motion": "infer_t2m",
        "temporal_motion_completion": "temporal_condition",
        "sequential_text_to_motion": "sequential_generation",
    },
    "motius.pipelines.t2mgpt.T2MGPTPipeline": {
        "text_to_motion": "infer_t2m",
    },
    "motius.pipelines.tm2t.TM2TPipeline": {
        "motion_to_text": "infer_m2t",
    },
    "motius.pipelines.unimumo.UniMuMoPipeline": {
        "music_to_dance": "infer_music_to_motion",
        "dance_to_music": "infer_motion_to_music",
    },
    "motius.pipelines.vermo.VermoPipeline": {
        "motion_to_text": "infer_m2t",
    },
    "motius.pipelines.vimogen.ViMoGenPipeline": {
        "text_to_motion": "infer_t2m",
    },
}

# These adapters intentionally change only the call shape. Their forwarding is
# covered by strict tests in tests/test_pipeline_task_api_parity.py.
VERIFIED_ADAPTER_ROUTES = {
    ("motius.pipelines.ardy.ARDYPipeline", "sequential_text_to_motion"),
    ("motius.pipelines.hymotion_t2m.HyMotionT2MPipeline", "text_to_motion"),
    ("motius.pipelines.kimodo.KIMODOPipeline", "text_to_motion"),
    ("motius.pipelines.kimodo.KIMODOPipeline", "sequential_text_to_motion"),
    ("motius.pipelines.kimodo.KIMODOPipeline", "temporal_motion_completion"),
}


def _import_object(path: str):
    module_name, _, object_name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), object_name)


def _discover_pipeline_classes() -> dict[str, str]:
    discovered = {}
    pipelines_root = REPO_ROOT / "motius" / "pipelines"
    for path in sorted(pipelines_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name.endswith("Pipeline")
            ):
                discovered[node.name] = str(path.relative_to(REPO_ROOT))
    return discovered


def audit_task_apis() -> dict:
    rows = []
    seen = set()
    bindings = [
        (spec.pipeline_class, task)
        for spec in CHECKPOINT_SPECS
        for task in spec.tasks
    ]
    bindings.extend(
        (pipeline_class, task)
        for pipeline_class, tasks in NON_HF_PIPELINE_TASKS.items()
        for task in tasks
    )
    for pipeline_class_path, task in bindings:
        key = (pipeline_class_path, task)
        if key in seen:
            continue
        seen.add(key)
        pipeline_class = _import_object(pipeline_class_path)
        canonical_name = f"infer_{task}"
        canonical = getattr(pipeline_class, canonical_name, None)
        legacy_name = LEGACY_TASK_ROUTES.get(pipeline_class_path, {}).get(
            task,
            canonical_name,
        )
        legacy = getattr(pipeline_class, legacy_name, None)
        errors = []
        if not callable(canonical):
            errors.append(f"missing {canonical_name}")
        if not callable(legacy):
            errors.append(f"missing legacy route {legacy_name}")

        if canonical_name == legacy_name:
            route_kind = "native"
        elif callable(canonical) and canonical is legacy:
            route_kind = "identity"
        elif key in VERIFIED_ADAPTER_ROUTES:
            route_kind = "verified_adapter"
        else:
            route_kind = "unverified_adapter"
            errors.append(
                f"{canonical_name} is not identical to {legacy_name} "
                "and has no strict adapter test"
            )
        rows.append(
            {
                "pipeline_class": pipeline_class_path,
                "task": task,
                "canonical_method": canonical_name,
                "legacy_entrypoint": legacy_name,
                "route_kind": route_kind,
                "status": "pass" if not errors else "fail",
                "errors": errors,
            }
        )

    discovered = _discover_pipeline_classes()
    audited_names = {
        row["pipeline_class"].rsplit(".", 1)[-1] for row in rows
    }
    accounted_names = audited_names | set(PIPELINE_CLASS_EXCLUSIONS)
    coverage_errors = [
        f"unclassified Pipeline class {name} in {path}"
        for name, path in discovered.items()
        if name not in accounted_names
    ]
    coverage_errors.extend(
        f"declared Pipeline class {name} was not found in motius/pipelines"
        for name in sorted(accounted_names)
        if name not in discovered
    )

    summary = {
        "artifacts": len(CHECKPOINT_SPECS),
        "artifact_task_bindings": sum(
            len(spec.tasks) for spec in CHECKPOINT_SPECS
        ),
        "source_pipeline_classes": len(discovered),
        "pipeline_classes": len({row["pipeline_class"] for row in rows}),
        "task_methods": len(rows),
        "native": sum(row["route_kind"] == "native" for row in rows),
        "identity": sum(row["route_kind"] == "identity" for row in rows),
        "verified_adapters": sum(
            row["route_kind"] == "verified_adapter" for row in rows
        ),
        "class_exclusions": len(PIPELINE_CLASS_EXCLUSIONS),
        "failed": (
            sum(row["status"] != "pass" for row in rows)
            + len(coverage_errors)
        ),
    }
    return {
        "summary": summary,
        "class_exclusions": [
            {
                "class_name": name,
                "reason": reason,
                "source": discovered.get(name),
            }
            for name, reason in sorted(PIPELINE_CLASS_EXCLUSIONS.items())
        ],
        "coverage_errors": coverage_errors,
        "task_methods": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audits/pipeline_task_api_parity.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_task_apis()
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    for row in report["task_methods"]:
        if row["status"] != "pass":
            print(
                f"{row['pipeline_class']}::{row['canonical_method']}: "
                + "; ".join(row["errors"])
            )
    for error in report["coverage_errors"]:
        print(error)
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
