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


@pytest.fixture
def assets(tmp_path):
    # Synthetic linear body, not an SMPL asset or a quality evaluation fixture.
    from motius.models.flowhmr.vendor.flowhmr.core.motion.motion_rep import _SMPLH_PARENTS
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


def test_training_backward_and_resume_counter(assets, config):
    torch.set_num_threads(2)
    bundle = FlowHMRBundle(config=config, **assets)
    trainer = FlowHMRTrainer(bundle)
    trainer.runner = SimpleNamespace(global_step=37)
    data = batch()
    original = deepcopy(data)
    out = trainer.train_step(data)
    assert bundle.model.global_iteration == 37
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
    first = bundle.training_forward(deepcopy(data))["loss"]
    torch.manual_seed(42)
    second = bundle.training_forward(changed)["loss"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_official_weights_and_export_roundtrip(assets, config, tmp_path):
    from motius import Pipeline
    bundle = FlowHMRBundle(config=config, **assets).eval()
    torch.save({"model_state_dict": bundle.model.state_dict()}, tmp_path / "official.ckpt")
    loaded = FlowHMRBundle(config=config, ckpt_path=tmp_path / "official.ckpt", **assets).eval()
    loaded.save_pretrained(tmp_path / "export")
    restored = Pipeline.from_pretrained(str(tmp_path / "export"), bundle_kwargs=assets).bundle
    feature = torch.randn(1,4,8)
    camera = torch.eye(3).reshape(1,1,9).expand(1,4,9)
    a = loaded.generate_from_feature(feature, camera)
    b = restored.generate_from_feature(feature, camera)
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
    torch.testing.assert_close(out["camera_R_condition"], torch.diag(torch.tensor([1.,-1.,-1.])).expand(4,3,3))
    with pytest.raises(ValueError, match="cfg_scale"):
        bundle.generate_from_feature(torch.zeros(1,4,8), torch.zeros(1,4,9), cfg_scale=0.5)


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
    state = {k: v for k,v in model.model.state_dict().items() if not _external_body_key(k)}
    path = tmp_path / "official.ckpt"
    torch.save({"model_state_dict": state}, path)
    model.load_checkpoint(path)
    state.pop("motion_transformer.input_encoder.weight")
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_checkpoint(path)


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
    args.update(mean_std=str(ASSETS / "motion_stats.json"),
                smpl_model_path=assets["body_model_path"], j_regressor_path=assets["j_regressor_path"])
    official = OfficialPipeline(network_module=config["network_module"],
                 network_module_args=deepcopy(config["network_module_args"]), **args)
    bundle = FlowHMRBundle(config=config, **assets)
    official.load_state_dict(bundle.model.state_dict(), strict=True)
    data = batch()
    torch.manual_seed(73)
    expected = official.forward_in_training(deepcopy(data))
    torch.manual_seed(73)
    actual = bundle.training_forward(deepcopy(data))
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
    for key in expected["loss_dict"]:
        torch.testing.assert_close(actual["loss_dict"][key], expected["loss_dict"][key], rtol=0, atol=0)
    actual["loss"].backward(); expected["loss"].backward()
    for (_,p), (_,q) in zip(bundle.model.named_parameters(), official.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    feature = {k:v[:1] for k,v in data["inputs"]["feature"].items()}
    official.eval(); bundle.eval()
    a = official.generate(deepcopy(feature), [13], 4)
    b = bundle.generate_from_feature(feature["feature"], feature["camera_R"], seeds=[13])
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
        weight = bundle.model.module.motion_transformer.final_layer.linear.weight.detach()
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
    torch.save({"model_state_dict": bundle.model.state_dict()}, checkpoint)
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
    for suffix in ("sam3d_body.data", "sam3d_body.data.utils",
                   "sam3d_body.data.transforms", "yolox.data", "vggt_omega.models"):
        assert "motius.models.flowhmr.vendor." + suffix in packages
