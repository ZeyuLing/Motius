"""Execute source-pinned GEM-X without network source imports or rendering."""

from __future__ import annotations

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


def _install_local_dinov3(root: Path) -> None:
    """Resolve official torch-hub calls against the pinned vendored DINOv3."""

    import torch

    original = torch.hub.load
    dino_root = root / "third_party" / "dinov3-repo"
    sys.path.insert(0, str(dino_root))

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).rstrip("/") != "facebookresearch/dinov3":
            return original(repo_or_dir, model, *args, **kwargs)
        kwargs.pop("source", None)
        kwargs.pop("skip_validation", None)
        # The fixed SAM3D caller uses the old alias. Drop-path is inactive in
        # eval mode, so translating it does not alter inference numerics.
        kwargs.pop("drop_path", None)
        from dinov3.hub import backbones

        constructor = getattr(backbones, model)
        return constructor(*args, **kwargs)

    torch.hub.load = load


def _install_deterministic_decode() -> None:
    """Move device-sensitive decode/integration operations to CPU for audits."""

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
            return self._torch.cumsum(value.cpu(), *args, **kwargs).to(value.device)

    motion_utils.torch = _CpuCumsumTorch(motion_utils.torch)
    postprocess.torch = _CpuCumsumTorch(postprocess.torch)


def main() -> None:
    root = Path(__file__).resolve().parent
    demo = root / "scripts" / "demo" / "demo_soma.py"
    sys.path[:0] = [
        str(demo.parent),
        str(root),
        str(root / "third_party" / "sam-3d-body"),
        str(root / "third_party" / "soma"),
    ]
    _install_local_dinov3(root)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    namespace = runpy.run_path(str(demo), run_name="_motius_gem_x_demo")
    seed = os.environ.get("MOTIUS_GEM_X_SEED")
    if seed is not None:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        if os.environ.get("MOTIUS_GEM_X_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            _install_deterministic_decode()
    demo_globals = namespace["main"].__globals__
    model_input_path = os.environ.get("MOTIUS_GEM_X_MODEL_INPUT_TRACE")
    if model_input_path:
        import torch

        official_load_data = demo_globals["load_data_dict"]

        def load_data_dict(cfg):
            data = official_load_data(cfg)
            output = Path(model_input_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_detach_cpu(data), output)
            return data

        demo_globals["load_data_dict"] = load_data_dict
    if os.environ.get("MOTIUS_GEM_X_SKIP_RENDER", "1") == "1":
        demo_globals["render_2d_keypoints"] = lambda *args, **kwargs: None
        demo_globals["render_incam"] = lambda *args, **kwargs: None
        demo_globals["render_global_o3d"] = lambda *args, **kwargs: None
        demo_globals["merge_videos_horizontal"] = lambda *args, **kwargs: None
    namespace["main"]()


if __name__ == "__main__":
    main()
