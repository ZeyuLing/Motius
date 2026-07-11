"""
tools/train.py — Main training entry point for motius.

Usage:
    python tools/train.py configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py [--work-dir WORK_DIR]
    # or with accelerate:
    accelerate launch tools/train.py configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bind_local_cuda_device():
    """Bind each distributed worker to its local GPU before model construction."""
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is None:
        return
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(int(local_rank) % device_count)
    except Exception as exc:
        print(
            f"Warning: failed to bind CUDA device for LOCAL_RANK={local_rank}: {exc}",
            file=sys.stderr,
            flush=True,
        )


_bind_local_cuda_device()

# Pre-import the HuggingFace `datasets` library before motius so that
# `motius.datasets` does not shadow it in sys.modules.
try:
    import datasets as _hf_datasets  # noqa: F401
except ImportError:
    pass

import motius  # noqa: E402
motius.register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with motius')
    parser.add_argument('config', help='Path to config file (.py)')
    parser.add_argument('--work-dir', '--work_dir', dest='work_dir',
                        help='Override work_dir in config')
    parser.add_argument('--auto-resume', '--auto_resume', dest='auto_resume',
                        action='store_true',
                        help='Auto-resume from latest checkpoint in work_dir')
    parser.add_argument('--load-from', '--load_from', dest='load_from',
                        help='Path to checkpoint to load from')
    parser.add_argument('--load-scope', '--load_scope', dest='load_scope',
                        default='model', choices=['model', 'full'],
                        help='Checkpoint load scope (model: weights only, full: full resume)')
    parser.add_argument('--cfg-options', '--cfg_options', dest='cfg_options', nargs='+',
                        help='Override config options, e.g. optimizer.lr=1e-4')
    # `accelerate launch` passes this. In plain single-process runs we should
    # leave LOCAL_RANK unset so Accelerate stays in non-distributed mode.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_rank is not None and 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        _bind_local_cuda_device()

    # Load config
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)

    # Apply CLI overrides
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.auto_resume:
        cfg.auto_resume = True
    if args.load_from:
        cfg.load_from = dict(path=args.load_from, load_scope=args.load_scope)
    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))

    # Derive work_dir from config filename unless explicitly overridden.
    # The base runtime sets work_dir='work_dirs/default' as a placeholder;
    # always replace it with a config-name-specific directory to prevent
    # cross-experiment checkpoint contamination during auto-resume.
    cfg_work_dir = getattr(cfg, 'work_dir', None) or ''
    if not args.work_dir and (not cfg_work_dir or cfg_work_dir == 'work_dirs/default'):
        cfg.work_dir = os.path.join(
            'work_dirs', os.path.splitext(os.path.basename(args.config))[0]
        )

    os.makedirs(cfg.work_dir, exist_ok=True)

    from motius.utils.logger import get_logger
    logger = get_logger()
    logger.info(f"Config: {args.config}")
    logger.info(f"Work dir: {cfg.work_dir}")

    # Build runner and train
    from motius.runner import AccelerateRunner
    try:
        runner = AccelerateRunner.from_cfg(cfg)
        runner.train()
    except Exception:
        logger.exception("Training failed with exception:")
        raise


def _parse_cfg_options(options):
    """Parse cfg_options like ['optimizer.lr=1e-4', 'train_cfg.max_iters=1000']."""
    result = {}
    for opt in options:
        key, val = opt.split('=', 1)
        try:
            import ast
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return result


if __name__ == '__main__':
    main()
