import json
from pathlib import Path

import torch
from mmengine import Config

from motius import Pipeline
from motius.models.animatecanvas import AnimateCanvasBundle
from motius.pipelines.animatecanvas import AnimateCanvasPipeline


ROOT = Path(__file__).resolve().parents[1]


def tiny_bundle():
    bundle = AnimateCanvasBundle(
        motion_transformer=dict(
            type='AnimateCanvasMMDiT',
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


def test_animatecanvas_artifact_roundtrip_and_auto_pipeline(tmp_path):
    source = tiny_bundle().eval()
    artifact = tmp_path / 'animatecanvas'
    source.save_pretrained(str(artifact), include_text_encoder=False)

    manifest = json.loads((artifact / 'model_index.json').read_text())
    assert manifest['pipeline_class'].endswith('AnimateCanvasPipeline')
    assert manifest['api']['inference'].endswith('infer_m2m')

    pipeline = Pipeline.from_pretrained(str(artifact), num_steps=1)
    assert isinstance(pipeline, AnimateCanvasPipeline)
    actual = pipeline.infer_m2m(num_frames=8, seed=17)
    assert actual['motion_198'].shape == (1, 8, 198)
    assert actual['lengths'] == [8]


def test_legacy_checkpoint_matches_renamed_artifact(tmp_path):
    """The rename must preserve inference from previously published metadata."""
    from motius.models.motioncanvas import MotionCanvasBundle
    from motius.pipelines.motioncanvas import MotionCanvasPipeline

    assert MotionCanvasBundle is AnimateCanvasBundle
    assert MotionCanvasPipeline is AnimateCanvasPipeline
    torch.manual_seed(23)
    artifact = tmp_path / 'artifact'
    tiny_bundle().eval().save_pretrained(str(artifact), include_text_encoder=False)
    canonical = Pipeline.from_pretrained(str(artifact), num_steps=2)
    expected = canonical.infer_m2m(num_frames=8, seed=17)

    for path in artifact.glob('*.json'):
        text = path.read_text()
        assert 'MotionCanvas' not in text and 'motioncanvas' not in text
        path.write_text(text.replace('AnimateCanvas', 'MotionCanvas').replace('animatecanvas', 'motioncanvas'))
    (artifact / 'animatecanvas_config.json').rename(artifact / 'motioncanvas_config.json')

    legacy = Pipeline.from_pretrained(str(artifact), num_steps=2, local_files_only=True)
    actual = legacy.infer_m2m(num_frames=8, seed=17)
    assert isinstance(legacy, AnimateCanvasPipeline)
    assert torch.equal(actual['motion_198'], expected['motion_198'])
    assert actual['lengths'] == expected['lengths']
    restored = AnimateCanvasBundle.from_config(artifact)
    assert isinstance(restored, AnimateCanvasBundle)


def test_animatecanvas_completion_preserves_known_coordinates():
    pipeline = AnimateCanvasPipeline(tiny_bundle().eval(), num_steps=1)
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


def test_animatecanvas_edit_context_matches_training_contract():
    pipeline = AnimateCanvasPipeline(tiny_bundle().eval(), num_steps=1)
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


def test_animatecanvas_training_config_uses_tracked_authority_assets():
    config = Config.fromfile(
        ROOT / 'configs/animatecanvas/train_animatecanvas_0p46b.py'
    )
    assert config.auto_resume is True
    assert config.work_dir.startswith('outputs/')
    assert config.model.mean_std_dir == 'checkpoints/models/animatecanvas'
    assert (
        config.model.bone_offsets_path
        == 'checkpoints/models/animatecanvas/bone_offsets_22.pt'
    )
    for name in ('Mean.npy', 'Std.npy', 'bone_offsets_22.pt'):
        assert (
            ROOT / 'checkpoints/models/animatecanvas' / name
        ).is_file()

    groups = config.train_dataloader.weighted_sampler.groups
    assert [group.frac for group in groups] == [0.3, 0.6, 0.05, 0.05]
    assert config.train_dataloader.weighted_sampler.num_samples == 630000
