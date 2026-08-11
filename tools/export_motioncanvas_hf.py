#!/usr/bin/env python3
"""Convert a converged MotionCanvas checkpoint to a Motius Hub artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.motioncanvas import MotionCanvasBundle


LEARNED_BUNDLE_TENSORS = (
    'null_vtxt_feat',
    'null_ctxt_input',
    'special_game_vtxt_feat',
    'special_game_ctxt_feat',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/motioncanvas/train_motioncanvas_0p46b.py'),
    )
    parser.add_argument('--stats-dir', type=Path, required=True)
    parser.add_argument('--bone-offsets', type=Path, required=True)
    parser.add_argument('--llm-path', type=Path, required=True)
    parser.add_argument('--sentence-encoder-path', type=Path, required=True)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/releases/motioncanvas_0p46b'),
    )
    parser.add_argument('--repo-id', default='ZeyuLing/Motius-MotionCanvas-0.46B')
    parser.add_argument(
        '--demo',
        action='append',
        default=[],
        help='Rendered MP4 to copy into the artifact; may be repeated.',
    )
    parser.add_argument(
        '--preview',
        action='append',
        default=None,
        help='Rendered GIF to copy into demos; defaults to release previews.',
    )
    parser.add_argument(
        '--model-card',
        type=Path,
        default=Path('configs/motioncanvas/huggingface_model_card.md'),
    )
    parser.add_argument('--upload', action='store_true')
    return parser.parse_args()


def load_source_checkpoint(path: Path):
    if path.is_dir():
        safe_path = path / 'model.safetensors'
        custom_path = path / 'custom_checkpoint_0.pkl'
        if safe_path.exists() and custom_path.exists():
            from safetensors.torch import load_file

            transformer = load_file(str(safe_path))
            learned = torch.load(
                custom_path,
                map_location='cpu',
                weights_only=True,
            )
            missing = [
                name for name in LEARNED_BUNDLE_TENSORS if name not in learned
            ]
            if missing:
                raise ValueError(
                    f'{custom_path} is missing learned bundle tensors: {missing}'
                )
            return transformer, learned, safe_path
        path = path / 'model.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'motion_transformer' not in payload:
        raise ValueError(f'{path} does not contain motion_transformer weights')
    transformer = payload['motion_transformer']
    learned = payload.get('__bundle_params__', {})
    missing = [name for name in LEARNED_BUNDLE_TENSORS if name not in learned]
    if missing:
        raise ValueError(f'{path} is missing learned bundle tensors: {missing}')
    return transformer, learned, path


def build_bundle(args, transformer_state, learned):
    model_cfg = dict(Config.fromfile(args.config).model)
    model_cfg.pop('type', None)
    model_cfg['motion_transformer'] = dict(model_cfg['motion_transformer'])
    model_cfg['motion_transformer']['type'] = 'MotionCanvasMMDiT'
    model_cfg['mean_std_dir'] = str(args.stats_dir)
    model_cfg['t2m_pretrained_path'] = None
    model_cfg['text_encoder'] = {
        'type': 'HYTextModel',
        'llm_type': 'qwen3',
        'llm_model_path': str(args.llm_path),
        'llm_tokenizer_path': str(args.llm_path),
        'max_length_llm': 128,
        'sentence_emb_type': 'clipl',
        'sentence_emb_model_path': str(args.sentence_encoder_path),
        'sentence_emb_tokenizer_path': str(args.sentence_encoder_path),
        'max_length_sentence_emb': 77,
        'enable_llm_padding': True,
    }
    bundle = MotionCanvasBundle(**model_cfg)
    bundle.motion_transformer.load_state_dict(transformer_state, strict=True)
    for name in LEARNED_BUNDLE_TENSORS:
        target = getattr(bundle, name)
        target.data.copy_(learned[name].to(dtype=target.dtype))

    mean = np.load(args.stats_dir / 'Mean.npy').astype(np.float32)
    std = np.load(args.stats_dir / 'Std.npy').astype(np.float32)
    for name, expected in (('mean', mean), ('std', std)):
        if name in learned:
            actual = learned[name].detach().cpu().numpy().astype(np.float32)
            if not np.array_equal(actual, expected):
                raise ValueError(f'checkpoint {name} differs from {args.stats_dir}')
    bundle.set_bone_offsets_override(
        torch.load(args.bone_offsets, map_location='cpu', weights_only=True)
    )
    return bundle


def write_release_metadata(args, source_checkpoint, transformer_state):
    metadata = {
        'format': 'motius-motioncanvas-source-audit-v1',
        'source_checkpoint': source_checkpoint.name,
        'source_epoch': 2354,
        'transformer_tensor_count': len(transformer_state),
        'training_config': args.config.name,
        'native_representation': 'MotionCanvas-198',
        'native_fps': 30,
    }
    (args.output_dir / 'source_checkpoint.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )
    stats_provenance = args.stats_dir / 'ANALYSIS.md'
    if stats_provenance.is_file():
        shutil.copy2(
            stats_provenance,
            args.output_dir / 'normalization_provenance.md',
        )


def copy_demos(args):
    demo_dir = args.output_dir / 'demos'
    demo_dir.mkdir(parents=True, exist_ok=True)
    for value in args.demo:
        source = Path(value)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, demo_dir / source.name)
    previews = args.preview
    if previews is None:
        previews = [
            'assets/model_zoo/motioncanvas/keyframe_control_512_30fps.gif',
            'assets/model_zoo/motioncanvas/trajectory_control_512_30fps.gif',
            'assets/model_zoo/motioncanvas/instruction_editing_512_30fps.gif',
            'assets/model_zoo/motioncanvas/motion_editing_512_30fps.gif',
        ]
    for value in previews:
        source = Path(value)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, demo_dir / source.name)


def copy_model_card(args):
    if not args.model_card.is_file():
        raise FileNotFoundError(args.model_card)
    shutil.copy2(args.model_card, args.output_dir / 'README.md')


def upload(args):
    from huggingface_hub import HfApi

    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN is required with --upload')
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type='model', exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type='model',
        folder_path=str(args.output_dir),
    )


def main():
    args = parse_args()
    if not str(args.output_dir).startswith('outputs/'):
        raise ValueError('--output-dir must be under outputs/')
    transformer_state, learned, source_checkpoint = load_source_checkpoint(
        args.checkpoint
    )
    bundle = build_bundle(args, transformer_state, learned)
    bundle.save_pretrained(
        str(args.output_dir),
        include_text_encoder=True,
        variant='0.46B-epoch2354',
    )
    write_release_metadata(args, source_checkpoint, transformer_state)
    copy_demos(args)
    copy_model_card(args)
    if args.upload:
        upload(args)
    print(args.output_dir)


if __name__ == '__main__':
    main()
