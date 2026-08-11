"""Execute the unmodified source-pinned GEM-SMPL demo from Motius."""

from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path


def _detach_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cpu(item) for item in value)
    return value


def main() -> None:
    root = Path(__file__).resolve().parent
    demo = root / "scripts" / "demo" / "demo_smpl_hpe.py"
    sys.path[:0] = [str(demo.parent), str(root)]
    namespace = runpy.run_path(str(demo), run_name="_motius_gem_smpl_demo")
    seed = os.environ.get("MOTIUS_GEM_SMPL_SEED")
    if seed is not None:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        if os.environ.get("MOTIUS_GEM_SMPL_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    model_input_path = os.environ.get("MOTIUS_GEM_SMPL_MODEL_INPUT_TRACE")
    if model_input_path:
        import torch

        demo_globals = namespace["main"].__globals__
        official_run_inference = demo_globals["run_inference"]

        def run_inference(model, data, static_cam):
            output = Path(model_input_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_detach_cpu(data), output)
            return official_run_inference(model, data, static_cam)

        demo_globals["run_inference"] = run_inference
    namespace["main"]()


if __name__ == "__main__":
    main()
