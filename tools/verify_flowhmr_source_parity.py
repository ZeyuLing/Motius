#!/usr/bin/env python3
"""Compare native FlowHMR with an independent upstream checkout at zero tolerance."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def exact(left, right):
    if left.keys() != right.keys():
        raise AssertionError(f'Output keys differ: {left.keys()} != {right.keys()}')
    keys = []
    for key in left:
        if torch.is_tensor(left[key]):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            keys.append(key)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frontend', type=Path,
                        help='Directory containing a verified frontend_parity.json and native.pt')
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(options.upstream.resolve()))
    from flowhmr.pipeline.pipeline_v2m import V2MPipeline
    from motius.models.flowhmr import FlowHMRBundle
    from motius.models.flowhmr.bundle import SOURCE_REVISION, _external_body_key
    from motius.pipelines.flowhmr import FlowHMRPipeline
    from motius.trainers.flowhmr import FlowHMRTrainer

    revision = subprocess.check_output(['git', '-C', str(options.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f'Expected upstream {SOURCE_REVISION}, got {revision}')
    options.output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(options.upstream)): digest(path)
                     for path in sorted((options.upstream / 'flowhmr').rglob('*.py'))}
    assets = dict(body_model_path=str(options.checkpoint_root / 'body_models/smplh/neutral/model.npz'),
                  j_regressor_path=str(options.checkpoint_root / 'body_models/smpl_neutral_J_regressor.pt'))
    reports = {'source_revision': revision, 'source_sha256': source_hashes,
               'device': options.device, 'torch_version': torch.__version__,
               'tolerance': {'atol': 0, 'rtol': 0}, 'variants': {}, 'complete': False}
    (options.output_dir / 'full_network_parity.json').write_text(json.dumps(reports, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    for variant in ('base', 'latest'):
        bundle = FlowHMRBundle.from_pretrained(options.checkpoint_root / f'flowhmr_{variant}',
                                              **assets, device=options.device)
        pipeline = FlowHMRPipeline(bundle)
        trainer = FlowHMRTrainer(bundle)
        cfg = copy.deepcopy(bundle.config)
        args = cfg['train_pipeline_args']
        args.update(smpl_model_path=assets['body_model_path'],
                    j_regressor_path=assets['j_regressor_path'], mean_std=str(bundle.mean_std_path))
        official = V2MPipeline(network_module=cfg['network_module'],
                              network_module_args=cfg['network_module_args'], **args).to(options.device)
        official_state = official.state_dict()
        native_state = bundle.export_state_dict()
        if set(official_state) - set(native_state) != {key for key in official_state if _external_body_key(key)}:
            raise AssertionError('Native export does not cover exactly the upstream network and statistics')
        official_state.update(native_state)
        official.load_state_dict(official_state, strict=True)
        del official_state, native_state
        entry = {'body_sha256': digest(assets['body_model_path']),
                 'checkpoint_sha256': bundle.checkpoint_sha256, 'training': [], 'sampling': []}
        feature = torch.randn(1, 4, 3072, device=options.device)
        camera = torch.eye(3, device=options.device).reshape(1, 1, 9).expand(1, 4, 9)
        if variant == 'base':
            rot = torch.tensor([1., 0., 0., 0., 1., 0.], device=options.device).expand(1, 4, 52, 6).clone()
            shapes = torch.zeros(1, 4, 16, device=options.device)
            trans = torch.zeros(1, 4, 3, device=options.device)
            with torch.no_grad():
                joints = pipeline.codec.body_model(dict(rot6d=rot.reshape(4, 52, 6),
                    shapes=shapes.reshape(4, 16), trans=trans.reshape(4, 3)))['keypoints3d'].reshape(1, 4, 52, 3)
            batch = dict(length=torch.tensor([4], device=options.device),
                         inputs={'feature': {'feature': feature, 'camera_R': camera}},
                         target=dict(smooth_root_vel=trans.clone(), smooth_root_pos=trans.clone(),
                                     local_joints_positions=joints, local_rot_data=rot,
                                     global_rot_data=rot.clone(), shapes=shapes,
                                     foot_contacts=torch.zeros(1, 4, 4, device=options.device)))
            for step in (1, 10001):
                bundle.train(); official.train(); official.global_iteration = step - 1
                torch.manual_seed(73)
                expected = official.forward_in_training(copy.deepcopy(batch))
                torch.manual_seed(73)
                actual = trainer.forward_batch(copy.deepcopy(batch), global_step=step)
                exact(expected['loss_dict'], actual['loss_dict'])
                torch.testing.assert_close(expected['loss'], actual['loss'], rtol=0, atol=0)
                expected['loss'].backward(); actual['loss'].backward()
                source_parameters = dict(official.motion_transformer.named_parameters())
                native_parameters = dict(bundle.network.named_parameters())
                assert source_parameters.keys() == native_parameters.keys()
                count = 0
                for name, source in source_parameters.items():
                    native = native_parameters[name]
                    assert (source.grad is None) == (native.grad is None), name
                    if source.grad is not None:
                        torch.testing.assert_close(source.grad, native.grad, rtol=0, atol=0)
                        count += 1
                entry['training'].append(dict(step=step, loss=actual['loss'].item(), exact_gradient_tensors=count))
                print(variant, 'training', entry['training'][-1], flush=True)
                official.zero_grad(set_to_none=True); trainer.zero_grad(set_to_none=True)
                del expected, actual
        bundle.eval(); official.eval()
        for seeds, scale in (((13,), 1.0), ((0, 19), 2.0)):
            expected = official.generate({'feature': feature, 'camera_R': camera}, list(seeds), 4, cfg_scale=scale)
            actual = pipeline.generate_from_feature(feature, camera, seeds=seeds, cfg_scale=scale)
            keys = exact(expected, actual)
            entry['sampling'].append(dict(frames=4, seeds=list(seeds), cfg_scale=scale, exact_keys=keys))
            print(variant, 'sampling', entry['sampling'][-1], flush=True)
            del expected, actual
        if options.frontend:
            proof = json.loads((options.frontend / 'frontend_parity.json').read_text())
            if not proof.get('complete') or proof.get('atol') != 0 or proof.get('rtol') != 0:
                raise ValueError('The supplied frontend has no complete zero-tolerance comparison')
            inputs = torch.load(options.frontend / 'native.pt', weights_only=True)
            source_inputs = torch.load(options.frontend / 'upstream.pt', weights_only=True)
            exact(source_inputs, inputs)
            from flowhmr.datasets.v2m_generation.base_dataset import compute_camera_features
            relative, _ = compute_camera_features(inputs['camera_RT'])
            count = len(inputs['features'])
            expected = official.generate({
                'feature': inputs['features'][None].to(options.device),
                'camera_R': relative.reshape(1, count, 9).to(options.device)}, [0], count)
            actual = pipeline.infer_from_feature(inputs['features'], inputs['camera_RT'], seeds=[0])
            torch.testing.assert_close(actual['camera_R_condition'], relative, rtol=0, atol=0)
            torch.testing.assert_close(actual['raw_trans'], expected['trans'].cpu(), rtol=0, atol=0)
            first = official.mesh_model(dict(rot6d=expected['rot6d'][:, 0],
                shapes=expected['shapes'][:, 0], trans=expected['trans'][:, 0]))
            floor = first['vertices'][..., 1].min(dim=-1).values
            expected['trans'] = expected['trans'].clone()
            expected['trans'][..., 1] -= floor[:, None]
            for name, value in expected.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual[name], value.cpu(), rtol=0, atol=0)
            params = dict(rot6d=expected['rot6d'].reshape(count, 52, 6),
                shapes=expected['shapes'].expand(1, count, -1).reshape(count, 16),
                trans=expected['trans'].reshape(count, 3))
            joints = official.body_model(params)['keypoints3d'].reshape(1, count, 52, 3)
            torch.testing.assert_close(actual['joints_world'], joints.cpu(), rtol=0, atol=0)
            entry['real_video'] = dict(frames=count, seed=0, conditioning='exact',
                sampling='exact', floor_alignment='exact', fk_joints='exact')
            print(variant, 'real video', entry['real_video'], flush=True)
            del expected, actual, joints
        reports['variants'][variant] = entry
        (options.output_dir / 'full_network_parity.json').write_text(json.dumps(reports, indent=2) + '\n')
        del trainer, pipeline, bundle, official
        gc.collect()
        if options.device.startswith('cuda'):
            torch.cuda.empty_cache()
    reports['complete'] = True
    tracked = ['motius/models/flowhmr', 'motius/trainers/flowhmr',
               'motius/motion/representation/flowhmr', 'motius/motion/body_models/flowhmr']
    paths = [path for directory in tracked for path in (root / directory).rglob('*.py')]
    paths.extend(root / name for name in ('motius/pipelines/flowhmr/pipeline.py',
        'motius/pipelines/flowhmr/sampling.py', 'motius/diffusion/flowhmr.py',
        'motius/diffusion/flowhmr_utils.py', 'motius/utils/module_loading.py',
        'motius/utils/flowhmr_checkpoint.py'))
    reports['native_sha256'] = {str(path.relative_to(root)): digest(path) for path in sorted(paths)}
    (options.output_dir / 'full_network_parity.json').write_text(json.dumps(reports, indent=2) + '\n')
    print('All source comparisons passed at zero tolerance.', flush=True)


if __name__ == '__main__':
    main()
