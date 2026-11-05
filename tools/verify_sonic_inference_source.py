"""Compare SONIC deployment composition, original ONNX and native artifacts."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from verify_sonic_network_source import original_roots
from verify_prompthmr_video_parity import tensors, compare_full


# Schema recovered from the public source-pinned ONNX Slice/Reshape nodes.
# It differs from the training schema while preserving every neural dimension.
DEPLOYMENT_SCHEMA = {
    'encoder_index': (3,),
    'command_multi_future_nonflat': (10, 58),
    'root_height_future': (10,),
    'root_height': (1,),
    'motion_anchor_ori_b': (6,),
    'motion_anchor_ori_b_mf_nonflat': (10, 6),
    'command_multi_future_lower_body': (240,),
    'vr_3point_local_target': (9,),
    'vr_3point_local_orn_target': (12,),
    'smpl_joints_multi_future_local_nonflat': (10, 72),
    'smpl_root_ori_b_multi_future': (10, 6),
    'joint_pos_multi_future_wrist_for_smpl': (10, 6),
}


def worker(args):
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.worker == 'cold':
        from motius import Pipeline as AutoPipeline

        pipeline = AutoPipeline.from_pretrained(
            args.output_dir / 'exported', bundle_kwargs={'device': 'cuda'}
        )
        inputs = torch.load(args.output_dir / 'export_input.pt', weights_only=True)
        torch.save(tensors(pipeline(**inputs)), args.output_dir / 'cold.pt')
        return
    checkpoint = args.artifact / 'sonic_release/last.pt'
    if args.worker == 'upstream':
        policy, _ = original_roots(
            args.upstream,
            checkpoint,
            checkpoint.parent / 'config.yaml',
            DEPLOYMENT_SCHEMA,
        )
        policy.eval()
    else:
        from motius.pipelines.sonic import SONICPipeline

        pipeline = SONICPipeline.from_original_checkpoint(checkpoint, device='cuda')
    results = {}
    for index, (batch, modes) in enumerate(
        [(1, [0]), (1, [1]), (1, [2]), (3, [0, 1, 2])]
    ):
        torch.manual_seed(613 + index)
        encoder = torch.randn(batch, 1762, device='cuda') * 0.13
        state = torch.randn(batch, 930, device='cuda') * 0.13
        encoder[:, 0] = torch.tensor(modes, device='cuda')
        encoder[:, 1:4] = torch.nn.functional.one_hot(encoder[:, 0].long(), 3).float()
        with torch.no_grad():
            if args.worker == 'upstream':
                result = policy.actor_module(
                    {'actor_obs': state[:, None], 'tokenizer': encoder[:, None, 1:]},
                    compute_aux_loss=False,
                    return_dict=True,
                )
                tokens = policy.actor_module.assemble_all_tokens(
                    result['encoded_tokens'], result['encoder_masks'], batch, 1
                )
                flat = tokens.flatten(start_dim=1)
                output = dict(
                    action=result['action_mean'].squeeze(1),
                    encoded_tokens=flat,
                    decoder_input=torch.cat([flat, state], -1),
                )
                if batch == 1:
                    import onnxruntime as ort

                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 4
                    session = ort.InferenceSession(
                        str(args.artifact / 'model_encoder.onnx'),
                        sess_options=options,
                        providers=['CPUExecutionProvider'],
                    )
                    original_tokens = torch.from_numpy(
                        session.run(None, {'obs_dict': encoder.cpu().numpy()})[0]
                    ).cuda()
                    # Quantization is discrete: require identical deployed tokens.
                    torch.testing.assert_close(original_tokens, flat, rtol=0, atol=0)
                    decoder = ort.InferenceSession(
                        str(args.artifact / 'model_decoder.onnx'),
                        sess_options=options,
                        providers=['CPUExecutionProvider'],
                    )
                    original_action = torch.from_numpy(
                        decoder.run(
                            None,
                            {'obs_dict': torch.cat([flat, state], -1).cpu().numpy()},
                        )[0]
                    ).cuda()
                    torch.testing.assert_close(
                        original_action, output['action'], rtol=2e-5, atol=2e-5
                    )
                    results['onnx/' + str(index)] = dict(
                        tokens_bitwise_equal=True,
                        decoder_max_abs_error=(original_action - output['action'])
                        .abs()
                        .max()
                        .item(),
                    )
            else:
                output = pipeline(encoder, state)
                output = {
                    key: torch.as_tensor(output[key], device='cuda')
                    for key in ['action', 'encoded_tokens', 'decoder_input']
                }
        results[str(index)] = tensors(output)
    if args.worker == 'native':
        # Current parameters must survive export and cold automatic loading.
        with torch.no_grad():
            pipeline.bundle.policy.decoders['g1_dyn'].module[-1].bias.add_(0.0125)
        pipeline.save_pretrained(args.output_dir / 'exported')
        inputs = dict(
            encoder_observation=encoder.cpu(), decoder_observation=state.cpu()
        )
        torch.save(inputs, args.output_dir / 'export_input.pt')
        torch.save(tensors(pipeline(**inputs)), args.output_dir / 'export_expected.pt')
    torch.save(results, args.output_dir / (args.worker + '.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'artifact', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', choices=['upstream', 'native', 'cold'])
    args = parser.parse_args()
    for name in ['upstream', 'artifact', 'output_dir']:
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'source_parity.json'
    report = dict(
        complete=False,
        scope='Actual original neural composition using the source-pinned public ONNX observation schema versus native Pipeline, all three modality modes and mixed batching. Original public CPU ONNX encoder tokens must be bitwise identical; CPU ONNX versus CUDA Torch decoder uses explicit 2e-5 tolerance. Modified current-weight export and cold AutoPipeline reload must be bitwise identical. No physical environment or training protocol certification.',
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        artifact=json.loads((args.artifact / 'provenance.json').read_text()),
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native', 'cold']:
        (args.output_dir / (kind + '.pt')).unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, '-u', __file__, *sys.argv[1:], '--worker', kind],
            cwd=args.upstream if kind == 'upstream' else root,
            env={
                **os.environ,
                'PYTHONPATH': str(root),
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            },
            check=True,
        )
    import torch

    torch.set_num_threads(4)
    original = torch.load(args.output_dir / 'upstream.pt', weights_only=True)
    report['original_onnx_checks'] = {
        key: original.pop(key) for key in list(original) if key.startswith('onnx/')
    }
    try:
        compare_full(
            original,
            torch.load(args.output_dir / 'native.pt', weights_only=True),
            'inference',
            report['tests'],
        )
        compare_full(
            torch.load(args.output_dir / 'export_expected.pt', weights_only=True),
            torch.load(args.output_dir / 'cold.pt', weights_only=True),
            'cold_export',
            report['tests'],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Original SONIC deployment composition and native cold artifact exact')


if __name__ == '__main__':
    main()
