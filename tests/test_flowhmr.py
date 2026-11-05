"""Real small-network gradients, checkpoint and camera tests without licensed assets."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from motius.models.flowhmr.bundle import ASSETS, FlowHMRBundle
from motius.trainers.flowhmr import FlowHMRTrainer
from motius.pipelines.flowhmr import FlowHMRPipeline


@pytest.fixture
def assets(tmp_path):
    # Synthetic linear body, not an SMPL asset or a quality evaluation fixture.
    from motius.motion.representation.flowhmr.motion import _SMPLH_PARENTS
    verts = 52
    np.savez(tmp_path / "body.npz", J_regressor=np.eye(52, dtype=np.float32),
             v_template=np.arange(verts*3, dtype=np.float32).reshape(verts,3)/100,
             shapedirs=np.zeros((verts,3,16), np.float32),
             posedirs=np.zeros((verts,3,51*9), np.float32),
             weights=np.eye(52, dtype=np.float32),
             kintree_table=_SMPLH_PARENTS.numpy(), f=np.array([[0,1,2]]))
    torch.save(torch.zeros(17, verts), tmp_path / "reg.pt")
    return dict(body_model_path=str(tmp_path / "body.npz"), j_regressor_path=str(tmp_path / "reg.pt"))


@pytest.fixture
def config():
    cfg = yaml.safe_load((ASSETS / "base_model.yml").read_text())
    cfg["network_module_args"].update(feat_dim=32, num_heads=2, num_layers=3,
                                    ctxt_input_dim={"feature": 8, "camera_R": 9}, vtxt_input_dim=8)
    cfg["network_module_args"]["text_refiner_cfg"] = {"num_layers": 1}
    cfg["train_pipeline_args"]["infer_noise_scheduler_cfg"]["validation_steps"] = 2
    return cfg


def batch():
    b,t = 2,4
    rot = torch.tensor([1.,0.,0.,0.,1.,0.]).expand(b,t,52,6).clone()
    return dict(length=torch.tensor([4,2]),
        inputs={"feature": {"feature": torch.randn(b,t,8), "camera_R": torch.eye(3).reshape(1,1,9).expand(b,t,9).clone()}},
        target={"smooth_root_vel": torch.zeros(b,t,3), "smooth_root_pos": torch.zeros(b,t,3),
                "local_joints_positions": torch.randn(b,t,52,3),
                "local_rot_data": rot, "global_rot_data": rot.clone(),
                "shapes": torch.zeros(b,t,16), "foot_contacts": torch.zeros(b,t,4)})


def test_lightweight_video_frontend_lifecycle_and_original_sampling(assets, config, tmp_path):
    """Only the execution path changes; camera/feature inputs and seeds stay fixed."""
    bundle = FlowHMRBundle(config=config, **assets)
    standard = FlowHMRPipeline(bundle)
    torch.manual_seed(11)
    features = torch.randn(4, 8)
    camera = torch.eye(4).repeat(4, 1, 1)
    expected = standard.infer_from_feature(features, camera, seeds=[3])
    light = FlowHMRPipeline(bundle, lightweight=True, attention_query_chunk_size=2)
    events = []

    class Preprocessor:
        def transcode_to_30fps(self, *args): return 'prepared.mp4'
        def detect_and_track(self, video): events.append('detection'); return torch.ones(4, 4), None
        def release_detector(self): events.append('release_detector')
        def extract_features(self, *args, **kwargs): events.append('features'); return features
        def release(self): events.append('release')

    actual = light.infer_v2m('input.mp4', preprocessor=Preprocessor(), work_dir=tmp_path,
                            camera_RT=camera, camera_K=torch.eye(3).repeat(4, 1, 1), seeds=[3])
    assert events == ['detection', 'release_detector', 'features', 'release']
    torch.testing.assert_close(actual['rot6d'], expected['rot6d'], rtol=1e-5, atol=1e-6)
    assert actual['inference_settings']['full_hand_features'] is True
    assert actual['inference_settings']['camera_frame_interval'] == 1


def test_training_backward_and_resume_counter(assets, config):
    torch.set_num_threads(2)
    bundle = FlowHMRBundle(config=config, **assets)
    trainer = FlowHMRTrainer(bundle)
    trainer.runner = SimpleNamespace(global_step=37)
    data = batch()
    original = deepcopy(data)
    out = trainer.train_step(data)
    assert trainer.global_iteration == 37
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in bundle.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert data["inputs"]["feature"]["feature"].shape == original["inputs"]["feature"]["feature"].shape


def test_padding_does_not_change_objective(assets, config):
    bundle = FlowHMRBundle(config=config, **assets)
    data = batch()
    changed = deepcopy(data)
    for tensor in changed["target"].values():
        tensor[1, 2:] += 7
    torch.manual_seed(42)
    first = FlowHMRTrainer(bundle).forward_batch(deepcopy(data))["loss"]
    torch.manual_seed(42)
    second = FlowHMRTrainer(bundle).forward_batch(changed)["loss"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_official_weights_and_export_roundtrip(assets, config, tmp_path):
    from motius import Pipeline
    bundle = FlowHMRBundle(config=config, **assets).eval()
    torch.save({"model_state_dict": bundle.export_state_dict()}, tmp_path / "official.ckpt")
    loaded = FlowHMRBundle(config=config, ckpt_path=tmp_path / "official.ckpt", **assets).eval()
    loaded.save_pretrained(tmp_path / "export")
    restored = Pipeline.from_pretrained(str(tmp_path / "export"), bundle_kwargs=assets).bundle
    feature = torch.randn(1,4,8)
    camera = torch.eye(3).reshape(1,1,9).expand(1,4,9)
    a = FlowHMRPipeline(loaded).generate_from_feature(feature, camera)
    b = FlowHMRPipeline(restored).generate_from_feature(feature, camera)
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert len(restored.checkpoint_sha256) == 64


def test_camera_conversion_and_multi_seed(assets, config):
    from motius.pipelines.flowhmr import FlowHMRPipeline
    bundle = FlowHMRBundle(config=config, **assets)
    pipe = FlowHMRPipeline(bundle)
    out = pipe.infer_from_feature(torch.randn(4,8), torch.eye(4), seeds=[0,1], ground_align=False)
    assert out["joints_world"].shape == (2,4,52,3)
    torch.testing.assert_close(out["camera_R_condition"], torch.eye(3).expand(4,3,3))
    with pytest.raises(ValueError, match="cfg_scale"):
        FlowHMRPipeline(bundle).generate_from_feature(torch.zeros(1,4,8), torch.zeros(1,4,9), cfg_scale=0.5)


def test_relative_camera_conditioning_is_world_axis_invariant(assets, config, monkeypatch):
    from motius.pipelines.flowhmr import FlowHMRPipeline
    bundle = FlowHMRBundle(config=config, **assets)
    pipe = FlowHMRPipeline(bundle)
    received = []
    generate = pipe.generate_from_feature

    def capture(feature, camera_r, **kwargs):
        received.append(camera_r.clone())
        return generate(feature, camera_r, **kwargs)

    monkeypatch.setattr(pipe, "generate_from_feature", capture)
    # Noncommuting rotations distinguish R_t @ R_0.T from reversed products.
    rx = torch.tensor([[1.,0.,0.], [0.,0.,-1.], [0.,1.,0.]])
    ry = torch.tensor([[0.,0.,1.], [0.,1.,0.], [-1.,0.,0.]])
    rt = torch.eye(4).repeat(4, 1, 1)
    rt[:, :3, :3] = torch.stack([ry, rx @ ry, ry @ rx @ ry, ry])
    expected = torch.stack([torch.eye(3), rx, ry @ rx, torch.eye(3)])
    feature = torch.randn(4, 8)
    out = pipe.infer_from_feature(feature, rt, ground_align=False)
    torch.testing.assert_close(received[0], expected.reshape(1, 4, 9))
    torch.testing.assert_close(out["camera_R_condition"], expected)
    torch.testing.assert_close(out["camera_RT"], rt)

    world_change = torch.eye(4)
    world_change[:3, :3] = rx @ ry
    world_change[:3, 3] = torch.tensor([1., 2., 3.])
    changed = pipe.infer_from_feature(feature, rt @ world_change, ground_align=False)
    torch.testing.assert_close(received[1], received[0])
    torch.testing.assert_close(changed["camera_R_condition"], expected)
    torch.testing.assert_close(changed["joints_world"], out["joints_world"])


def test_config_registration():
    from mmengine.config import Config
    from motius.registry import MODEL_BUNDLES, TRAINERS, DATASETS
    import motius.datasets.flowhmr  # noqa
    cfg = Config.fromfile(Path(__file__).parents[1] / "configs/flowhmr/train_flowhmr.py")
    assert MODEL_BUNDLES.get(cfg.model.type) is FlowHMRBundle
    assert TRAINERS.get(cfg.trainer.type) is FlowHMRTrainer
    assert DATASETS.get(cfg.train_dataloader.dataset.type) is not None
    assert cfg.accelerator.mixed_precision == "no"


def test_official_checkpoint_omits_only_body_assets(assets, config, tmp_path):
    from motius.models.flowhmr.bundle import _external_body_key
    model = FlowHMRBundle(config=config, **assets)
    state = {k: v for k,v in model.export_state_dict().items() if not _external_body_key(k)}
    path = tmp_path / "official.ckpt"
    torch.save({"model_state_dict": state}, path)
    model.load_checkpoint(path)
    state.pop("motion_transformer.input_encoder.weight")
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_checkpoint(path)


def test_bundle_is_independent_of_licensed_body_assets(config, assets):
    bundle = FlowHMRBundle(config=config)
    assert not hasattr(bundle.model, "body_model")
    assert not hasattr(bundle, "generate_from_feature")
    assert not hasattr(bundle, "training_forward")
    with pytest.raises(ValueError, match="body_model_path"):
        FlowHMRPipeline(bundle)
    pipeline = FlowHMRPipeline(bundle, body_model_path=assets["body_model_path"])
    output = pipeline.infer_from_feature(torch.randn(4, 8), torch.eye(4))
    assert output["joints_world"].shape == (1, 4, 52, 3)


def test_runner_resume_restores_representation_statistics(assets, config):
    source = FlowHMRBundle(config=config, **assets)
    source.motion_statistics["smooth_root_pos_mean"].add_(0.25)
    checkpoint = deepcopy(source.state_dict_to_save())
    restored = FlowHMRBundle(config=config, **assets)
    # The trainer can already exist when the runner restores a checkpoint.
    trainer = FlowHMRTrainer(restored)
    restored.load_state_dict_selective(checkpoint, strict=True)
    trainer.train_step(batch())
    torch.testing.assert_close(trainer.codec.smooth_root_pos_mean,
                               source.motion_statistics["smooth_root_pos_mean"], rtol=0, atol=0)


def test_official_shard_preprocessing_and_backward(assets, config, tmp_path):
    import io
    import tarfile
    from motius.datasets.flowhmr import FlowHMRWebDataset
    from torch.utils.data import DataLoader
    t = 12
    fields = {
        "motion.npz": dict(poses=np.zeros((t,52,3), np.float32),
                           trans=np.tile([0.,0.,3.], (t,1)).astype(np.float32),
                           betas=np.zeros((1,16), np.float32)),
        "camera.npz": dict(RT=np.tile(np.eye(4), (t,1,1)).astype(np.float32),
                           K=np.tile(np.diag([500.,500.,1.]), (t,1,1)).astype(np.float32)),
        "bbox.npz": dict(bbox=np.tile([0.,0.,200.,200.], (t,1)).astype(np.float32),
                         start_end=np.array([0,t-1])),
    }
    path = tmp_path / "sample.tar"
    with tarfile.open(path, "w") as tar:
        for name, data in fields.items():
            buf = io.BytesIO(); np.savez(buf, **data)
            payload = buf.getvalue()
            info = tarfile.TarInfo("sample." + name); info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        buf = io.BytesIO(); torch.save(torch.randn(t,8), buf)
        info = tarfile.TarInfo("sample.feature.pt"); info.size = len(buf.getvalue())
        tar.addfile(info, io.BytesIO(buf.getvalue()))
    ds = FlowHMRWebDataset(str(path), assets["body_model_path"], max_len=16,
                           resampled=False, shuffle_buffer=1, cfg_dropout_prob=0.)
    data = next(iter(DataLoader(ds, batch_size=1)))
    assert data["length"].item() == t
    assert data["target"]["local_rot_data"].shape == (1,16,52,6)
    bundle = FlowHMRBundle(config=config, **assets)
    out = FlowHMRTrainer(bundle).train_step(data)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_against_upstream_forward_and_sampling(assets, config, monkeypatch):
    """Opt-in source parity, independent upstream classes using identical weights."""
    import os
    upstream = os.environ.get("MOTIUS_FLOWHMR_UPSTREAM")
    if not upstream:
        pytest.skip("Set MOTIUS_FLOWHMR_UPSTREAM to the pinned official checkout")
    monkeypatch.syspath_prepend(upstream)
    from flowhmr.pipeline.pipeline_v2m import V2MPipeline as OfficialPipeline
    args = deepcopy(config["train_pipeline_args"])
    args.update(mean_std=str(FlowHMRBundle(config=config, **assets).mean_std_path),
                smpl_model_path=assets["body_model_path"], j_regressor_path=assets["j_regressor_path"])
    official = OfficialPipeline(network_module=config["network_module"],
                 network_module_args=deepcopy(config["network_module_args"]), **args)
    bundle = FlowHMRBundle(config=config, **assets)
    official_state = official.state_dict()
    native_state = bundle.export_state_dict()
    from motius.models.flowhmr.bundle import _external_body_key
    assert set(official_state) - set(native_state) == {k for k in official_state if _external_body_key(k)}
    official_state.update(native_state)
    official.load_state_dict(official_state, strict=True)
    data = batch()
    torch.manual_seed(73)
    expected = official.forward_in_training(deepcopy(data))
    torch.manual_seed(73)
    actual = FlowHMRTrainer(bundle).forward_batch(deepcopy(data))
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
    for key in expected["loss_dict"]:
        torch.testing.assert_close(actual["loss_dict"][key], expected["loss_dict"][key], rtol=0, atol=0)
    actual["loss"].backward(); expected["loss"].backward()
    for (_,p), (_,q) in zip(bundle.model.named_parameters(), official.motion_transformer.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    feature = {k:v[:1] for k,v in data["inputs"]["feature"].items()}
    official.eval(); bundle.eval()
    a = official.generate(deepcopy(feature), [13], 4)
    b = FlowHMRPipeline(bundle).generate_from_feature(feature["feature"], feature["camera_R"], seeds=[13])
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def _ddp_worker(rank, rendezvous, assets, config):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(1)
        bundle = FlowHMRBundle(config=config, **assets)
        bundle.model = DistributedDataParallel(bundle.model)
        trainer = FlowHMRTrainer(bundle)
        optimizer = torch.optim.AdamW(bundle.parameters(), lr=1e-3)
        # Different local data must still produce identical updated weights.
        for step in range(2):
            torch.manual_seed(rank + 20 + step)
            optimizer.zero_grad()
            trainer.train_step(batch())["loss"].backward()
            optimizer.step()
        weight = bundle.model.module.final_layer.linear.weight.detach()
        received = [torch.empty_like(weight) for _ in range(2)]
        dist.all_gather(received, weight)
        torch.testing.assert_close(received[0], received[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_process_training_sync(assets, config, tmp_path):
    torch.multiprocessing.spawn(_ddp_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", assets, config), nprocs=2, join=True)


def test_video_adapter_and_shared_capture(assets, config, tmp_path, monkeypatch):
    import cv2
    from motius.pipelines.flowhmr import FlowHMRPipeline
    from motius.motion.representation.monocular_capture import save_monocular_capture_result
    video = str(tmp_path / "input.mp4")
    # The front end is injected; this contract test must not depend on the
    # host's optional video encoder. Network, mesh, and result export are real.
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: SimpleNamespace(
        get=lambda prop: 30., release=lambda: None))
    bundle = FlowHMRBundle(config=config, **assets)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"model_state_dict": bundle.export_state_dict()}, checkpoint)
    bundle.load_checkpoint(checkpoint)
    intrinsic = torch.diag(torch.tensor([500.,500.,1.])).expand(4,3,3)

    class Frontend:
        def detect_and_track(self, path):
            return torch.tensor([0.,0.,32.,32.]).expand(4,4), None

        def extract_features(self, path, boxes, cameras, **kwargs):
            torch.testing.assert_close(cameras, intrinsic)
            return torch.zeros(4,8)

    pipeline = FlowHMRPipeline(bundle)
    result = pipeline.infer_monocular_motion_capture(video,
        work_dir=str(tmp_path / "run"), preprocessor=Frontend(), transcode=False,
        camera_RT=torch.eye(4), camera_K=intrinsic)
    assert result.tracks[0].joints_world.shape == (4,22,3)
    assert result.tracks[0].joints_camera is None
    assert result.camera_to_world is None
    assert result.output_fps == 30
    assert result.tracks[0].shape_parameters.shape == (1,16)
    save_monocular_capture_result(result, tmp_path / "capture.npz")
    with np.load(tmp_path / "capture.npz", allow_pickle=False) as saved:
        assert saved.files


def test_frontend_is_included_in_distribution():
    from setuptools import find_packages
    root = Path(__file__).resolve().parents[1]
    packages = set(find_packages(str(root), include=["motius*"]))
    for package in ("motius.pipelines.flowhmr.sam_data",
                    "motius.pipelines.flowhmr.sam_data.utils",
                    "motius.pipelines.flowhmr.sam_data.transforms",
                    "motius.pipelines.flowhmr.yolox_data",
                    "motius.models.vision.vggt_omega.models"):
        assert package in packages


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float16])
def test_meta_inference_load_materializes_exact_parameters_and_rope(assets, config, tmp_path, storage_dtype):
    original = FlowHMRBundle(config=config, **assets).eval().to(dtype=storage_dtype)
    original.save_pretrained(tmp_path/'inference')
    eager = FlowHMRBundle.from_pretrained(tmp_path/'inference', low_cpu_mem_usage=False, **assets).eval()
    meta = FlowHMRBundle.from_pretrained(tmp_path/'inference', low_cpu_mem_usage=True, **assets).eval()
    for (name,a),(other,b) in zip(eager.named_parameters(),meta.named_parameters()):
        assert name == other and not b.is_meta
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for (name,a),(other,b) in zip(eager.named_buffers(),meta.named_buffers()):
        assert name == other and not b.is_meta
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    data=batch()['inputs']['feature']
    kwargs=dict(x=torch.randn(2,4,491),ctxt_input=data,vtxt_input=torch.zeros(2,1,8),
                timesteps=torch.ones(2),x_mask_temporal=torch.ones(2,4,dtype=torch.bool),
                ctxt_mask_temporal=torch.ones(2,4,dtype=torch.bool))
    with torch.no_grad():
        torch.testing.assert_close(eager.predict_motion(**kwargs),meta.predict_motion(**kwargs),rtol=0,atol=0)


def test_vggt_camera_only_skips_depth_without_changing_pose():
    from torch import nn
    from motius.models.vision.vggt_omega.models.vggt_omega import VGGTOmega
    from motius.models.vision.vggt_omega.models.heads.camera_head import CameraHead
    class Aggregator(nn.Module):
        depth=2
        def forward(self, images, *, cached_layer_indices=None, progress=None):
            tokens=images.mean((-1,-2)).mean(-1)[...,None,None].expand(-1,-1,2,64)
            return [tokens if cached_layer_indices is None else None,tokens],1
    class Depth(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError('camera-only extraction must not evaluate depth')
    model=VGGTOmega.__new__(VGGTOmega);nn.Module.__init__(model)
    model.autocast=False;model.aggregator=Aggregator();model.camera_head=CameraHead(dim_in=64)
    model.dense_head=None;model.text_alignment_head=None;model.eval()
    images=torch.randn(2,3,8,8)
    with torch.no_grad():
        reference=model(images)['pose_enc']
        model.dense_head=Depth();actual=model(images,camera_only=True)
    assert 'depth' not in actual
    torch.testing.assert_close(reference,actual['pose_enc'],rtol=0,atol=0)


def test_vggt_meta_load_preserves_real_camera_prediction(tmp_path):
    from motius.models.vision.vggt_omega.models.vggt_omega import VGGTOmega
    from motius.models.vision.vggt_omega.bundle import VGGTOmegaBundle
    config=dict(embed_dim=64, enable_depth=False, autocast=False)
    source=VGGTOmega(**config).eval()
    checkpoint=tmp_path/'vggt.pt'
    torch.save(source.state_dict(),checkpoint)
    eager=VGGTOmegaBundle(checkpoint,low_cpu_mem_usage=False,**config).eval().model
    meta=VGGTOmegaBundle(checkpoint,**config).eval().model
    images=torch.randn(1,2,3,16,16)
    with torch.no_grad():
        reference=eager(images)['pose_enc']
        actual=meta(images,camera_only=True)['pose_enc']
    torch.testing.assert_close(reference,actual,rtol=0,atol=0)


def test_fast_sam_backend_selection_is_explicit(assets,config):
    bundle=FlowHMRBundle(config=config,**assets)
    fast=FlowHMRPipeline(bundle,lightweight=True)
    assert fast.preprocessor_kwargs['sam_fast'] is True
    assert fast.preprocessor_kwargs['sam_token_only'] is True
    official=FlowHMRPipeline(bundle,lightweight=True,sam_backend='official')
    assert official.preprocessor_kwargs['sam_fast'] is False
    assert official.preprocessor_kwargs['sam_token_only'] is True
    assert official.preprocessor_kwargs['sam_frame_batch_size'] == 8
    legacy=FlowHMRPipeline(bundle,lightweight=True,preprocessor_kwargs={'sam_token_only':False})
    assert legacy.preprocessor_kwargs['sam_fast'] is False
    with pytest.raises(ValueError,match='sam_backend'):
        FlowHMRPipeline(bundle,sam_backend='unknown')
