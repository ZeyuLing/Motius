#!/usr/bin/env python3
"""Cold-load one Motius Hub artifact and reject secondary model downloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _blocked_download(*_args, **_kwargs):
    raise AssertionError("secondary Hugging Face download attempted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache/hf_full_direct_load"),
    )
    parser.add_argument(
        "--bundle-kwargs",
        default="{}",
        help="JSON object forwarded to the artifact bundle constructor.",
    )
    parser.add_argument("--revision")
    parser.add_argument("--token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_kwargs = json.loads(args.bundle_kwargs)
    if not isinstance(bundle_kwargs, dict):
        raise TypeError("--bundle-kwargs must decode to a JSON object")

    started = time.monotonic()
    artifact = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
            token=args.token,
        )
    )

    from motius import Pipeline

    metadata = Pipeline.resolve_pretrained(artifact)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import huggingface_hub

    huggingface_hub.snapshot_download = _blocked_download
    huggingface_hub.hf_hub_download = _blocked_download
    try:
        import clip.clip
    except ImportError:
        pass
    else:
        clip.clip._download = _blocked_download

    pipeline = Pipeline.from_pretrained(
        artifact,
        bundle_kwargs=bundle_kwargs,
    )
    task_methods = [f"infer_{task}" for task in metadata.tasks]
    missing = [
        method for method in task_methods if not callable(getattr(pipeline, method, None))
    ]
    if missing:
        raise AssertionError(f"missing task methods after load: {missing}")

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": metadata.revision,
                "pipeline_class": pipeline.__class__.__name__,
                "task_methods": task_methods,
                "artifact": str(artifact),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "status": "pass",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
