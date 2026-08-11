import json
from pathlib import Path

import torch
from mmengine import Config

from motius import Pipeline
from motius.models.motioncanvas import MotionCanvasBundle
from motius.pipelines.motioncanvas import MotionCanvasPipeline


ROOT = Path(__file__).resolve().parents[1]


def tiny_bundle():
    bundle = MotionCanvasBundle(
        motion_transformer=dict(
            type='MotionCanvasMMDiT',
            input_dim=594,
            output_dim=198,
            feat_dim=64,
            num_heads=4,
            num_layers=3,
            dropout=0.0,
            max_motion_len=360,
            ctxt_input_dim=4096,
            vtxt_input_dim=768,
            text_refiner_cfg=dict(num_layers=1),
        ),
        losses_cfg={},
        uncondition_mode=True,
    )
    bundle.mean = torch.zeros(198)
    bundle.std = torch.ones(198)
    bundle.set_bone_offsets_override(torch.zeros(22, 3))
    return bundle


def test_motioncanvas_artifact_roundtrip_and_auto_pipeline(tmp_path):
    source = tiny_bundle().eval()
    artifact = tmp_path / 'motioncanvas'
    source.save_pretrained(str(artifact), include_text_encoder=False)

    manifest = json.loads((artifact / 'model_index.json').read_text())
    assert manifest['pipeline_class'].endswith('MotionCanvasPipeline')
    assert manifest['api']['inference'].endswith('infer_m2m')

    pipeline = Pipeline.from_pretrained(str(artifact), num_steps=1)
    assert isinstance(pipeline, MotionCanvasPipeline)
    actual = pipeline.infer_m2m(num_frames=8, seed=17)
    assert actual['motion_198'].shape == (1, 8, 198)
    assert actual['lengths'] == [8]


def test_motioncanvas_completion_preserves_known_coordinates():
    pipeline = MotionCanvasPipeline(tiny_bundle().eval(), num_steps=1)
    source = torch.randn(1, 8, 198)
    generate = torch.zeros(1, 8)
    generate[:, 3:6] = 1.0

    result = pipeline.infer_temporal_motion_completion(
        source,
        generate,
        seed=9,
    )
    output = result['motion_198']
    known = ~generate.bool()
    assert torch.equal(output[known], source[known])


def test_motioncanvas_edit_context_matches_training_contract():
    pipeline = MotionCanvasPipeline(tiny_bundle().eval(), num_steps=1)
    source = torch.randn(1, 8, 198)
    edit = source + 2.0
    generate = torch.zeros(1, 8)
    generate[:, 2:4] = 1.0
    mask = pipeline._as_generation_mask(generate, source)

    completion_source = source * (1.0 - mask)
    editing_source = source * (1.0 - mask) + edit * mask
    completion_context = pipeline.bundle.prepare_condition_context(
        completion_source,
        src_mask=mask,
    )
    editing_context = pipeline.bundle.prepare_condition_context(
        editing_source,
        src_mask=mask,
    )

    assert torch.count_nonzero(completion_context[..., :198]) == 0
    assert torch.equal(editing_context[..., :198], edit * mask)


def test_motioncanvas_training_config_uses_tracked_authority_assets():
    config = Config.fromfile(
        ROOT / 'configs/motioncanvas/train_motioncanvas_0p46b.py'
    )
    assert config.auto_resume is True
    assert config.work_dir.startswith('outputs/')
    assert config.model.mean_std_dir == 'checkpoints/models/motioncanvas'
    assert (
        config.model.bone_offsets_path
        == 'checkpoints/models/motioncanvas/bone_offsets_22.pt'
    )
    for name in ('Mean.npy', 'Std.npy', 'bone_offsets_22.pt'):
        assert (
            ROOT / 'checkpoints/models/motioncanvas' / name
        ).is_file()

    groups = config.train_dataloader.weighted_sampler.groups
    assert [group.frac for group in groups] == [0.3, 0.6, 0.05, 0.05]
    assert config.train_dataloader.weighted_sampler.num_samples == 630000
