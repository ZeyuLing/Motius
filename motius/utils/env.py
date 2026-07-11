"""Environment info utilities."""

import platform
import sys


def collect_env_info() -> str:
    """Collect environment information for debugging."""
    lines = []
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {sys.version}")

    try:
        import torch
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA version: {torch.version.cuda}")
            lines.append(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        lines.append("PyTorch: not installed")

    for pkg in ['accelerate', 'transformers', 'diffusers', 'peft', 'mmengine', 'safetensors']:
        try:
            mod = __import__(pkg)
            lines.append(f"{pkg}: {mod.__version__}")
        except ImportError:
            lines.append(f"{pkg}: not installed")

    return '\n'.join(lines)
