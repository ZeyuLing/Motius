#!/usr/bin/env python3
"""Render compact multi-case Model Zoo gallery assets.

The script consumes already exported ``motion_135``/SMPL-parameter NPZ files
from the internal evaluation output tree and writes compact GIF previews for
the public model cards. It intentionally avoids running model inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio.v2 as imageio
from PIL import Image

from render_motion135_smpl_demo import SMPLRenderer, load_motion


CASES = {
    "001840": "someone executes a roundhouse kick with their left foot.",
    "004545": "a person jumping while raising both hands and moving apart legs.",
    "006944": "a person moves their right hand left, right, up, and down.",
    "014457": "the person swings a golf club.",
    "M013344": "in a fighting stance, person punches downward with their right hand.",
}
DEFAULT_CASES = ("001840", "004545", "006944")

MODELS = {
    "mdm": {
        "method": "MDM",
        "source": "mdm",
        "prefix": "mdm",
        "cases": ("001840", "014457", "006944"),
    },
    "t2mgpt": {
        "method": "T2M-GPT",
        "source": "t2mgpt",
        "prefix": "t2mgpt",
    },
    "momask": {
        "method": "MoMask",
        "source": "momask",
        "prefix": "momask",
    },
    "mogents": {
        "method": "MoGenTS",
        "source": "mogents",
        "prefix": "mogents",
    },
    "motiongpt": {
        "method": "MotionGPT",
        "source": "motiongpt",
        "prefix": "motiongpt",
    },
    "flowmdm": {
        "method": "FlowMDM",
        "source": "flowmdm",
        "prefix": "flowmdm",
    },
    "motionmillion": {
        "method": "MotionMillion-7B",
        "source": "gotozero_7b_train",
        "prefix": "motionmillion_7b_train",
    },
    "motionstreamer": {
        "method": "MotionStreamer",
        "source": "motionstreamer",
        "prefix": "motionstreamer",
    },
    "hymotion_t2m": {
        "method": "HY-Motion T2M",
        "source": "hymotion_1b",
        "prefix": "hymotion_t2m_full",
    },
    "kimodo": {
        "method": "KIMODO",
        "source": "kimodo",
        "prefix": "kimodo",
    },
    "mld": {
        "method": "MLD",
        "source": "mld",
        "prefix": "mld",
        "cases": ("001840", "M013344", "006944"),
    },
    "motionlcm": {
        "method": "MotionLCM",
        "source": "motionlcm",
        "prefix": "motionlcm",
    },
    "vimogen": {
        "method": "ViMoGen",
        "source": "vimogen_1_3b_deepseek_caption",
        "prefix": "vimogen_1_3b_prompt_rewrite",
    },
    "dart": {
        "method": "DART",
        "source": "dart",
        "prefix": "dart",
    },
    "condmdi": {
        "method": "CondMDI",
        "source": "motion135",
        "prefix": "condmdi",
        "cases": ("014457", "001840", "006944"),
        "captions": {
            "014457": "the person swings a golf club.",
            "001840": "hands in fighting position while the left foot kicks aggressively up and over.",
            "006944": "the person who does arms straight out and then it's doing something with their right hand in front of their face.",
        },
    },
}


def gif_frame_durations_ms(frame_count: int, fps: int) -> list[int]:
    """Quantize frame times to GIF centiseconds without changing average fps."""
    if fps <= 0 or fps > 100:
        raise ValueError("GIF fps must be in the range 1..100")
    ticks = [round(index * 100 / fps) for index in range(frame_count + 1)]
    return [10 * max(1, ticks[index + 1] - ticks[index]) for index in range(frame_count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(
            "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/"
            "outputs/evaluation/t2m/humanml3d_official_test/motion135"
        ),
    )
    parser.add_argument("--out-root", type=Path, default=Path("assets/model_zoo"))
    parser.add_argument("--model-dir", type=Path, default=Path("ref_repo/MDM/body_models"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--write-mp4", action="store_true", help="Also write MP4 sources next to GIF previews.")
    parser.add_argument("--models", nargs="*", default=sorted(MODELS))
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Override each selected model's default preview cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    renderer = SMPLRenderer(args.model_dir, args.device, args.width, args.height)
    manifest_path = args.out_root / "gallery_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        manifest: dict[str, dict[str, object]] = dict(payload.get("models", {}))
        case_catalog = dict(payload.get("cases", {}))
    else:
        manifest = {}
        case_catalog = {}
    case_catalog.update(CASES)

    for model_key in args.models:
        info = MODELS[model_key]
        model_out = args.out_root / model_key
        model_out.mkdir(parents=True, exist_ok=True)
        rendered_cases = []
        case_ids = args.cases if args.cases is not None else info.get("cases", DEFAULT_CASES)
        for case_id in case_ids:
            source = args.eval_root / str(info["source"]) / f"{case_id}.npz"
            if not source.exists():
                raise FileNotFoundError(source)
            name = f"{info['prefix']}_humanml3d_{case_id}_smpl_mesh"
            global_orient, body_pose, transl, meta = load_motion(source)
            frames = renderer.render(
                renderer.vertices(global_orient, body_pose, transl),
                args.fps,
                args.max_frames,
            )
            gif = model_out / f"{name}_{args.width}_{args.fps}fps.gif"
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=gif_frame_durations_ms(len(frames), args.fps),
                loop=0,
                disposal=2,
            )
            caption = info.get("captions", CASES)[case_id]
            case_meta = {
                **meta,
                "method": info["method"],
                "sample_id": case_id,
                "caption": caption,
                "gif": str(gif),
                "frames": len(frames),
                "fps": args.fps,
                "width": args.width,
                "height": args.height,
            }
            if args.write_mp4:
                mp4 = model_out / f"{name}.mp4"
                imageio.mimwrite(mp4, frames, fps=args.fps, quality=8, macro_block_size=1)
                case_meta["mp4"] = str(mp4)
            (model_out / f"{name}.json").write_text(json.dumps(case_meta, indent=2) + "\n")
            rendered_cases.append(
                {
                    "sample_id": case_id,
                    "caption": caption,
                    "gif": str(gif),
                    "metadata": str(model_out / f"{name}.json"),
                }
            )
            print(f"rendered {model_key}/{case_id}: {gif}")
        manifest[model_key] = {
            "method": info["method"],
            "cases": rendered_cases,
        }

    manifest_path.write_text(
        json.dumps({"cases": case_catalog, "models": manifest}, indent=2) + "\n"
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
