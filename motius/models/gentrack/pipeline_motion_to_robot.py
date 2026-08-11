#!/usr/bin/env python3
"""End-to-end pipeline: HyMotion eval output -> Robot motion cache (V6: PyRoki).

Chains three conversion steps using PyRoki trajectory-level retargeting:
  1. motion_135 NPZ -> PyRoki keypoints .npy  (motion135_to_pyroki_keypoints.py)
  2. PyRoki keypoints -> Retargeted robot NPZ  (batch_retarget_to_g1_from_keypoints.py)
  3. Retargeted NPZ -> ProtoMotions .motion    (convert_pyroki_retargeted_robot_motions_to_proto.py)

V6 replaces the V5 GMR frame-by-frame IK pipeline with PyRoki's trajectory-level
optimization (jaxls LeastSquaresProblem, 800 iterations) which jointly solves:
  - Local bone alignment (weight=1.0)
  - Global keypoint alignment (weight=4.0)
  - Foot contact cost (weight=30.0)
  - Joint smoothness (weight=4.0)
  - Root smoothness (weight=1.0)
  - Joint velocity limit (weight=50.0)

Usage:
    python tools/gentrack/pipeline_motion_to_robot.py \\
        --input outputs/evaluation/gentrack/candidates/00000.npz \\
        --output outputs/evaluation/gentrack/robot_motion/ \\
        [--smpl-model-path checkpoints/smpl_models] \\
        [--keep-intermediates]

Optionally run ONNX tracker validation:
    python tools/gentrack/pipeline_motion_to_robot.py \\
        --input outputs/evaluation/gentrack/candidates/00000.npz \\
        --output outputs/evaluation/gentrack/robot_motion/ \\
        --validate
"""
import argparse
import os
import sys
import pathlib
import subprocess
import tempfile
import shutil

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
from motius.models.gentrack.tracker_paths import PROTOMOTIONS_ROOT

# PyRoki scripts in ProtoMotions
PYROKI_RETARGET_SCRIPT = PROTOMOTIONS_ROOT / "pyroki" / "batch_retarget_to_g1_from_keypoints.py"
PROTO_CONVERT_SCRIPT = PROTOMOTIONS_ROOT / "data" / "scripts" / "convert_pyroki_retargeted_robot_motions_to_proto.py"

# Default paths
DEFAULT_SMPL_MODEL = PROJECT_ROOT / "checkpoints" / "smpl_models"
DEFAULT_URDF = PROTOMOTIONS_ROOT / "protomotions" / "data" / "assets" / "urdf" / "for_retargeting" / "g1.urdf"
DEFAULT_ONNX = PROTOMOTIONS_ROOT / "data" / "pretrained_models" / "motion_tracker" / "g1-bones-deploy" / "compiled_models" / "unified_pipeline.onnx"


def run_step(cmd, step_name, cwd=None, env_extra=None):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*60}")
    print(f"  Step: {step_name}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=False,  # let output stream to console
        env=env,
    )
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"\n  {step_name} completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: HyMotion eval output -> Robot motion (PyRoki V6)"
    )
    parser.add_argument("--input", required=True,
                        help="Input motion_135 NPZ from HyMotion eval")
    parser.add_argument("--output", required=True,
                        help="Output directory for ProtoMotions .motion file")
    parser.add_argument("--smpl-model-path", default=str(DEFAULT_SMPL_MODEL),
                        help="Path to SMPL model directory (default: checkpoints/smpl_models)")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF),
                        help="Path to G1 URDF for retargeting")
    parser.add_argument("--fps", type=int, default=30,
                        help="Motion FPS (default: 30)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate files (keypoints .npy, retargeted .npz)")
    parser.add_argument("--validate", action="store_true",
                        help="Run ONNX tracker validation after conversion")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX),
                        help="Path to ONNX model for validation")
    # PyRoki retargeting options
    parser.add_argument("--subsample-factor", type=int, default=1,
                        help="Subsample factor for PyRoki (default: 1, no subsampling)")
    parser.add_argument("--target-raw-frames", type=int, default=450,
                        help="Target raw frames for PyRoki (default: 450)")
    # ProtoMotions convert options
    parser.add_argument("--output-fps", type=int, default=None,
                        help="Output FPS for ProtoMotions .motion. Defaults to fps/subsample-factor.")
    parser.add_argument("--robot-type", default="g1",
                        help="Robot type (default: g1)")
    parser.add_argument("--pyroki-max-iterations", type=int,
                        default=int(os.environ.get("PYROKI_MAX_ITERATIONS", "800")),
                        help="Maximum PyRoki/JAXLS solver iterations (default: 800; lower for smoke tests)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).resolve()
    output_dir = pathlib.Path(args.output).resolve()

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    effective_output_fps = (
        args.output_fps
        if args.output_fps is not None
        else max(1, int(round(args.fps / max(args.subsample_factor, 1))))
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine intermediate file paths
    stem = input_path.stem
    if args.keep_intermediates:
        inter_dir = output_dir / "intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        keypoints_dir = inter_dir / "keypoints"
        retarget_dir = inter_dir / "retargeted"
        contacts_dir = inter_dir / "contacts"
    else:
        tmpdir = tempfile.mkdtemp(prefix="embodied_pipeline_v6_")
        keypoints_dir = pathlib.Path(tmpdir) / "keypoints"
        retarget_dir = pathlib.Path(tmpdir) / "retargeted"
        contacts_dir = pathlib.Path(tmpdir) / "contacts"

    # Create subdirectories
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    retarget_dir.mkdir(parents=True, exist_ok=True)

    keypoints_path = keypoints_dir / f"{stem}.npy"

    try:
        # ==================================================================
        # Step 1: motion_135 -> PyRoki keypoints
        # ==================================================================
        run_step([
            sys.executable, SCRIPT_DIR / "motion135_to_pyroki_keypoints.py",
            str(input_path),
            str(keypoints_path),
            "--smpl-model-path", str(args.smpl_model_path),
            "--fps", str(args.fps),
        ], "motion_135 -> PyRoki keypoints (.npy)")

        if not keypoints_path.exists():
            print(f"ERROR: Keypoints file not created: {keypoints_path}")
            sys.exit(1)

        # ==================================================================
        # Step 2a: Extract foot contact labels (save-contacts-only mode)
        # ==================================================================
        contacts_cmd = [
            sys.executable, str(PYROKI_RETARGET_SCRIPT),
            "--save-contacts-only",
            "--keypoints-folder-path", str(keypoints_dir),
            "--contacts-dir", str(contacts_dir),
            "--source-type", "smpl",
            "--subsample-factor", str(args.subsample_factor),
            "--target-raw-frames", str(args.target_raw_frames),
            "--input-fps", str(args.fps),
            "--max-iterations", str(args.pyroki_max_iterations),
        ]
        run_step(contacts_cmd, "Extract foot contact labels")

        # ==================================================================
        # Step 2b: PyRoki retargeting (keypoints -> robot NPZ)
        # ==================================================================
        # batch_retarget_to_g1_from_keypoints.py takes a FOLDER of .npy files
        retarget_cmd = [
            sys.executable, str(PYROKI_RETARGET_SCRIPT),
            "--no-visualize",
            "--keypoints-folder-path", str(keypoints_dir),
            "--output-dir", str(retarget_dir),
            "--urdf-path", str(args.urdf),
            "--source-type", "smpl",
            "--subsample-factor", str(args.subsample_factor),
            "--target-raw-frames", str(args.target_raw_frames),
            "--input-fps", str(args.fps),
            "--max-iterations", str(args.pyroki_max_iterations),
        ]
        run_step(retarget_cmd, "PyRoki retargeting (keypoints -> robot NPZ)",
                 env_extra={"PYROKI_MAX_ITERATIONS": str(args.pyroki_max_iterations)})

        # Verify retargeted file exists
        retargeted_npz = retarget_dir / f"{stem}.npz"
        if not retargeted_npz.exists():
            # Try alternate naming patterns
            retargeted_files = list(retarget_dir.glob("*.npz"))
            if retargeted_files:
                print(f"  Found retargeted file(s): {[f.name for f in retargeted_files]}")
            else:
                print(f"ERROR: No retargeted NPZ found in {retarget_dir}")
                sys.exit(1)

        # ==================================================================
        # Step 3: Retargeted NPZ -> ProtoMotions .motion
        # ==================================================================
        proto_cmd = [
            sys.executable, str(PROTO_CONVERT_SCRIPT),
            "--retargeted-motion-dir", str(retarget_dir),
            "--output-dir", str(output_dir),
            "--input-fps", str(args.fps),
            "--output-fps", str(effective_output_fps),
            "--robot-type", str(args.robot_type),
            "--force-remake",  # Always overwrite existing .motion files to ensure contacts are applied
        ]
        # Add contacts dir if contacts were saved
        if contacts_dir.exists() and any(contacts_dir.iterdir()):
            proto_cmd += ["--contact-labels-dir", str(contacts_dir)]
        # motion_filter.py lives alongside the convert script in data/scripts/
        data_scripts_dir = str(PROTO_CONVERT_SCRIPT.parent)
        pythonpath_extra = os.environ.get("PYTHONPATH", "")
        if data_scripts_dir not in pythonpath_extra:
            pythonpath_extra = f"{data_scripts_dir}:{pythonpath_extra}" if pythonpath_extra else data_scripts_dir
        run_step(proto_cmd, "Retargeted NPZ -> ProtoMotions .motion",
                 cwd=str(PROTOMOTIONS_ROOT),
                 env_extra={"PYTHONPATH": pythonpath_extra, "MUJOCO_GL": "disable"})

        # ==================================================================
        # Done!
        # ==================================================================
        print(f"\n{'='*60}")
        print(f"  Pipeline complete! (PyRoki V6)")
        print(f"{'='*60}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_dir}")

        # List output files
        output_files = list(output_dir.glob("*.motion")) + list(output_dir.glob("*.pt"))
        for f in output_files:
            print(f"    -> {f}")

        if args.keep_intermediates:
            print(f"  Keypoints: {keypoints_dir}")
            print(f"  Retargeted: {retarget_dir}")

        # ==================================================================
        # Step 4 (optional): ONNX tracker validation
        # ==================================================================
        if args.validate:
            motion_files = list(output_dir.glob("*.motion"))
            if not motion_files:
                print("\nWARNING: No .motion files found for validation")
            elif not pathlib.Path(args.onnx).exists():
                print(f"\nWARNING: ONNX model not found: {args.onnx}")
                print("  Skipping validation.")
            else:
                tracker_script = PROTOMOTIONS_ROOT / "deployment" / "test_tracker_mujoco.py"
                for motion_file in motion_files:
                    run_step([
                        sys.executable, str(tracker_script),
                        "--onnx", str(args.onnx),
                        "--motion", str(motion_file),
                        "--loops", "1",
                        "--no-realtime",
                    ], f"ONNX Tracker Validation ({motion_file.name})")

    finally:
        if not args.keep_intermediates:
            # Clean up temp files
            if 'tmpdir' in locals() and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
