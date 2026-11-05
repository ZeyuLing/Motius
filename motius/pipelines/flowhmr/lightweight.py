"""Resource and licensed-asset resolution for FlowHMR video inference."""
from contextlib import contextmanager
import gc
import os
from pathlib import Path

import torch


def resolve_body_model(explicit=None):
    """Find the required neutral SMPL-H asset; never substitute another gender."""
    configured = explicit or os.environ.get('MOTIUS_SMPLH_MODEL')
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'FlowHMR SMPL-H model not found: {path}')
        return str(path)
    roots = []
    if os.environ.get('MOTIUS_BODY_MODEL_DIR'):
        roots.append(Path(os.environ['MOTIUS_BODY_MODEL_DIR']).expanduser())
    for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
        roots.extend((root / 'checkpoints/body_models', root / 'outputs/checkpoints/flowhmr/body_models'))
    for root in roots:
        for relative in ('smplh/neutral/model.npz', 'smplh/SMPLH_NEUTRAL.npz'):
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(
        'FlowHMR requires the licensed neutral SMPL-H model with 16 shape coefficients. '
        'Set MOTIUS_SMPLH_MODEL or put it in checkpoints/body_models/smplh/neutral/model.npz. '
        'Male/female models are not interchangeable with the neutral model.')


def release_allocator(device):
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def release_preprocessor(preprocessor, method):
    release = getattr(preprocessor, method, None)
    if callable(release):
        release()


@contextmanager
def frontend_resources(pipeline, preprocessor, device):
    """Keep motion weights on CPU while frontends execute, including on errors."""
    if not pipeline.lightweight:
        yield
        return
    modules = (pipeline.bundle, pipeline.codec)
    try:
        for module in modules:
            module.to('cpu')
        release_allocator(device)
        yield
    finally:
        release_preprocessor(preprocessor, 'release')
        for module in modules:
            module.to(device)
        release_allocator(device)
