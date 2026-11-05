"""Original CSV resampling and deployment exporter versus native interfaces.

Executes pristine source class bodies; only simulator-owned reference data is
supplied externally. Does not claim physical FK or pretrained-policy validation.
"""
from __future__ import annotations
import argparse, ast, copy, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from verify_prompthmr_video_parity import tensors, compare_full

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def source_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'rsl-rl', 'isaaclab', 'checkpoint', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    import numpy as np
    import onnxruntime as ort
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, scope='Original CSV interpolation/velocity math and original PyTorch/ONNX deployment exporter; seeded original G1 neural checkpoint with synthetic external reference data, not physical FK or released pretrained policy.', tests=[])
    report_path = args.output_dir / 'source_parity.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    def check(original, native, name):
        compare_full(tensors(original), tensors(native), name, report['tests'])
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    math_path = args.isaaclab / 'source/isaaclab/isaaclab/utils/math.py'
    spec = importlib.util.spec_from_file_location('original_isaac_math', math_path)
    math_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = math_module
    spec.loader.exec_module(math_module)
    namespace = dict(torch=torch, np=np, **{name: getattr(math_module, name) for name in
        ['axis_angle_from_quat', 'quat_conjugate', 'quat_mul', 'quat_slerp']})
    original_loader = source_class(args.upstream / 'scripts/csv_to_npz.py', 'MotionLoader', namespace)
    from motius.motion.representation.beyondmimic import BeyondMimicCSVReference
    rng = np.random.default_rng(617)
    csv = rng.normal(size=(120, 36)).astype(np.float32)
    csv[:, 3:7] /= np.linalg.norm(csv[:, 3:7], axis=-1, keepdims=True)
    csv_path = args.output_dir / 'reference.csv'
    np.savetxt(csv_path, csv, delimiter=',')
    for input_fps, output_fps, frame_range in [(30, 50, None), (60, 25, (12, 92)), (50, 50, (1, 100))]:
        original = original_loader(str(csv_path), input_fps, output_fps, torch.device('cpu'), frame_range)
        native = BeyondMimicCSVReference(str(csv_path), input_fps, output_fps, torch.device('cpu'), frame_range)
        label = f'csv/{input_fps}/{output_fps}/{frame_range}'
        check({key: value for key, value in vars(original).items() if isinstance(value, torch.Tensor)},
              {key: value for key, value in vars(native).items() if isinstance(value, torch.Tensor)}, label)
        for step in range(original.output_frames + 2):
            check(original.get_next_state(), native.get_next_state(), label + f'/step/{step}')

    sys.path.insert(0, str(args.rsl_rl))
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization
    from motius.models.beyondmimic import BeyondMimicPolicyBundle
    from motius.pipelines.beyondmimic import BeyondMimicPipeline
    from motius.pipelines.beyondmimic.export import BeyondMimicDeploymentGraph, export_onnx_policy
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    policy = ActorCritic(160, 286, 29, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation='elu')
    policy.load_state_dict(state['model_state_dict'], strict=True)
    norm = EmpiricalNormalization([160], until=1e8)
    norm.load_state_dict(state['obs_norm_state_dict'], strict=True)
    # Use nontrivial running statistics, shared by the actual source and native layers.
    torch.manual_seed(618)
    norm(torch.randn(19, 160) * 1.3 + .2)
    norm.eval()
    bundle = BeyondMimicPolicyBundle.from_original_checkpoint(args.checkpoint).eval()
    bundle.actor_normalizer.load_state_dict(norm.state_dict(), strict=True)
    reference = {name: rng.normal(size=shape).astype(np.float32) for name, shape in {
        'joint_pos': (17, 29), 'joint_vel': (17, 29), 'body_pos_w': (17, 14, 3),
        'body_quat_w': (17, 14, 4), 'body_lin_vel_w': (17, 14, 3), 'body_ang_vel_w': (17, 14, 3)}.items()}
    source_namespace = dict(torch=torch, copy=copy, os=os, ManagerBasedRLEnv=object)
    base = source_class(args.isaaclab / 'source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py', '_OnnxPolicyExporter', source_namespace)
    source_namespace['_OnnxPolicyExporter'] = base
    exporter_cls = source_class(args.upstream / 'source/whole_body_tracking/whole_body_tracking/utils/exporter.py', '_OnnxMotionPolicyExporter', source_namespace)
    command = SimpleNamespace(motion=SimpleNamespace(**{name: torch.from_numpy(value) for name, value in reference.items()}))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    source = exporter_cls(env, policy, norm).eval()
    native = BeyondMimicDeploymentGraph(bundle, reference).eval()
    pipeline = BeyondMimicPipeline(bundle, reference_motion=reference, control_hz=25,
                                  metadata=dict(joint_names=['joint_' + str(i) for i in range(29)]))
    obs = torch.randn(1, 160)
    names = ['actions', *reference]
    for step in [-1, 0, 7, 16, 100]:
        time_step = torch.tensor([[step]], dtype=torch.float32)
        with torch.no_grad():
            expected = source(obs, time_step)
            check(expected, native(obs, time_step), f'deployment/{step}')
        check(dict(zip(names, expected)), {name: pipeline(obs, step)[name] for name in names}, f'pipeline/{step}')
    source.export(str(args.output_dir), 'source.onnx')
    export_onnx_policy(bundle, reference, args.output_dir / 'native.onnx')
    sessions = [ort.InferenceSession(str(args.output_dir / name), providers=['CPUExecutionProvider']) for name in ['source.onnx', 'native.onnx']]
    for step in [-1, 0, 7, 100]:
        inputs = dict(obs=obs.numpy(), time_step=np.array([[step]], dtype=np.float32))
        check(sessions[0].run(None, inputs), sessions[1].run(None, inputs), f'onnx/{step}')
    # Save currently modified parameters and nondefault timing, then cold-load the artifact.
    with torch.no_grad():
        bundle.network.actor[-1].bias.add_(.03125)
    saved = args.output_dir / 'exported'
    pipeline.save_pretrained(saved)
    from motius import Pipeline
    restored = Pipeline.from_pretrained(saved)
    check(bundle.state_dict(), restored.bundle.state_dict(), 'artifact/current_weights')
    for step in [-1, 7, 100]:
        check(pipeline(obs, step), restored(obs, step), f'artifact/pipeline/{step}')
    assert restored.control_hz == 25
    assert restored.metadata == pipeline.metadata
    report.update(complete=True, torch_version=torch.__version__)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print('Original CSV, PyTorch/ONNX deployment and current-weight artifact comparisons are exact')


if __name__ == '__main__':
    main()
