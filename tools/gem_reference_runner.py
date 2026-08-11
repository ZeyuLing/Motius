"""Development-only runner for comparing pinned official GEM checkouts."""

from __future__ import annotations

import argparse
import os
import runpy
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


def _seed_from_env(method: str) -> None:
    import torch

    key = f"MOTIUS_{method.upper().replace('-', '_')}_SEED"
    seed = os.environ.get(key)
    if seed is None:
        return
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    deterministic_key = f"MOTIUS_{method.upper().replace('-', '_')}_DETERMINISTIC"
    if os.environ.get(deterministic_key) == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if method == "gem_x":
            from gem.network import endecoder
            from gem.pipeline import postprocess
            from gem.utils import motion_utils

            official_decode_soma_v2 = endecoder.EnDecoder.decode_soma_v2

            def decode_soma_v2(self, x_norm):
                device = x_norm.device
                decoded = official_decode_soma_v2(self, x_norm.cpu())
                return {key: value.to(device) for key, value in decoded.items()}

            endecoder.EnDecoder.decode_soma_v2 = decode_soma_v2

            def ensure_soma_model(self):
                if self.soma_model is None:
                    self.soma_model = endecoder.SomaLayer(
                        data_root="inputs/soma_assets",
                        low_lod=True,
                        device="cuda",
                        identity_model_type="mhr",
                        mode="torch",
                    )

            endecoder.EnDecoder._ensure_soma_model = ensure_soma_model
            official_matrix_to_axis_angle = postprocess.matrix_to_axis_angle

            def matrix_to_axis_angle(value):
                return official_matrix_to_axis_angle(value.cpu()).to(value.device)

            postprocess.matrix_to_axis_angle = matrix_to_axis_angle

            class _CpuCumsumTorch:
                def __init__(self, torch_module):
                    self._torch = torch_module

                def __getattr__(self, name):
                    return getattr(self._torch, name)

                def cumsum(self, value, *args, **kwargs):
                    if not value.is_cuda:
                        return self._torch.cumsum(value, *args, **kwargs)
                    return self._torch.cumsum(
                        value.cpu(), *args, **kwargs
                    ).to(value.device)

            motion_utils.torch = _CpuCumsumTorch(motion_utils.torch)
            postprocess.torch = _CpuCumsumTorch(postprocess.torch)


def _run_gem_smpl(source_root: Path, arguments: list[str]) -> None:
    import torch

    demo = source_root / "scripts/demo/demo_smpl_hpe.py"
    sys.path[:0] = [str(demo.parent), str(source_root)]
    namespace = runpy.run_path(str(demo), run_name="_motius_reference_gem_smpl")
    _seed_from_env("gem_smpl")
    demo_globals = namespace["main"].__globals__
    official_run_inference = demo_globals["run_inference"]
    model_input_path = Path(os.environ["MOTIUS_GEM_SMPL_MODEL_INPUT_TRACE"])

    def run_inference(model, data, static_cam):
        model_input_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_detach_cpu(data), model_input_path)
        return official_run_inference(model, data, static_cam)

    demo_globals["run_inference"] = run_inference
    sys.argv = [str(demo), *arguments]
    namespace["main"]()


def _run_gem_x(source_root: Path, arguments: list[str]) -> None:
    import torch

    demo = source_root / "scripts/demo/demo_soma.py"
    sys.path[:0] = [
        str(demo.parent),
        str(source_root),
        str(source_root / "third_party/sam-3d-body"),
        str(source_root / "third_party/soma"),
        str(source_root / "third_party/dinov3-repo"),
    ]
    namespace = runpy.run_path(str(demo), run_name="_motius_reference_gem_x")
    _seed_from_env("gem_x")
    demo_globals = namespace["main"].__globals__
    official_load_data = demo_globals["load_data_dict"]
    model_input_path = Path(os.environ["MOTIUS_GEM_X_MODEL_INPUT_TRACE"])

    def load_data_dict(cfg):
        data = official_load_data(cfg)
        model_input_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_detach_cpu(data), model_input_path)
        return data

    demo_globals["load_data_dict"] = load_data_dict
    demo_globals["render_2d_keypoints"] = lambda *args, **kwargs: None
    demo_globals["render_incam"] = lambda *args, **kwargs: None
    demo_globals["render_global_o3d"] = lambda *args, **kwargs: None
    demo_globals["merge_videos_horizontal"] = lambda *args, **kwargs: None
    sys.argv = [str(demo), *arguments]
    namespace["main"]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gem-smpl", "gem-x"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    source_root = args.source_root.expanduser().resolve()
    if args.method == "gem-smpl":
        _run_gem_smpl(source_root, remaining)
    else:
        _run_gem_x(source_root, remaining)


if __name__ == "__main__":
    main()
