"""Low-memory VGGT retains whole-sequence attention and original frame order."""
import pytest
import torch
from motius.models.vision.vggt_omega.models.layers.block import SelfAttentionBlock
from motius.models.vision.vggt_omega.models.aggregator import Aggregator
from motius.pipelines.flowhmr.camera_estimation import _StagedCameraModules


def test_camera_precision_constructs_and_caches_distinct_models(tmp_path, monkeypatch):
    from motius.pipelines.flowhmr import camera_estimation
    import motius.models.vision.vggt_omega.bundle as module
    calls = []
    class Stub:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.model = self
        def to(self, device): return self
        def eval(self): return self
    monkeypatch.setattr(module, 'VGGTOmegaBundle', Stub)
    monkeypatch.setattr(camera_estimation, '_VGGT_MODEL_CACHE', {})
    checkpoint = tmp_path/'camera.pt'
    checkpoint.touch()
    default = camera_estimation._load_vggt_model(str(checkpoint), 'cpu', precision='source')
    half = camera_estimation._load_vggt_model(str(checkpoint), 'cpu', precision='float16')
    full = camera_estimation._load_vggt_model(str(checkpoint), 'cpu', precision='float32')
    assert camera_estimation._load_vggt_model(str(checkpoint), 'cpu', precision='float16') is half
    assert len({id(default), id(half), id(full)}) == 3
    assert calls == [{'autocast': True}, {'autocast': True, 'autocast_dtype': 'float16'}, {'autocast': False}]


@pytest.mark.parametrize('qk_norm', [False, True])
@pytest.mark.parametrize('chunk', [1, 7, 64])
def test_chunked_global_attention_matches_complete_keys(qk_norm, chunk):
    torch.manual_seed(13)
    block = SelfAttentionBlock(64, 4, qkv_bias=True, init_values=.5,
                              use_qk_norm=qk_norm, mask_k_bias=True).eval()
    x = torch.randn(2, 31, 64)
    with torch.inference_mode():
        expected = block(x)
        actual = block.forward_chunked_(x.clone(), query_chunk_size=chunk, ffn_chunk_size=11)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('frame_batch', [1, 2, 8])
def test_streamed_camera_prefix_preserves_batches_frames_and_register_attention(frame_batch):
    torch.manual_seed(17)
    model = Aggregator(embed_dim=64, depth=3, register_attention_block_indices=(1,),
                       cached_layer_indices=(2,)).eval()
    # Activate residuals to expose errors hidden by the tiny initialization.
    with torch.no_grad():
        for block in [*model.frame_blocks, *model.inter_frame_blocks]:
            block.ls1.gamma.fill_(.25)
            block.ls2.gamma.fill_(.25)
    images = torch.randn(2, 5, 3, 32, 32)
    seen = []
    hook = model.patch_embed.register_forward_pre_hook(lambda module, args: seen.append(len(args[0])))
    with torch.inference_mode():
        expected, prefix = model(images)
        seen.clear()
        actual, actual_prefix = model.forward_camera_streamed(images, frame_batch_size=frame_batch,
                                                              token_chunk_size=13, query_chunk_size=7)
    hook.remove()
    assert max(seen) == min(frame_batch,10) and sum(seen) == 10
    assert actual_prefix == prefix
    assert actual[-1].shape[2] == prefix
    torch.testing.assert_close(actual[-1], expected[-1][:, :, :prefix], rtol=3e-5, atol=5e-6)


def test_staging_cleanup_runs_on_failure():
    class Module:
        def __init__(self): self.moves = []
        def to(self, device): self.moves.append(str(device))
    module = Module()
    def fail(): raise ValueError('inference failed')
    with pytest.raises(ValueError, match='inference failed'):
        _StagedCameraModules('cpu')(module, fail)
    assert module.moves == ['cpu', 'cpu']


def test_inplace_chunking_rejects_training_or_gradients():
    block = SelfAttentionBlock(16, 2)
    with torch.no_grad(), pytest.raises(RuntimeError, match='eval'):
        block.forward_chunked_(torch.randn(1, 4, 16))
    block.eval()
    with pytest.raises(RuntimeError, match='disabled gradients'):
        block.forward_chunked_(torch.randn(1, 4, 16))


def test_parallel_vggt_image_preparation_preserves_pixels_order_and_padding(tmp_path):
    import numpy as np
    from PIL import Image
    from motius.pipelines.flowhmr.vggt_utils.load_fn import load_and_preprocess_images
    rng = np.random.default_rng(71)
    paths = []
    for index, (h, w) in enumerate([(73, 131), (127, 67), (31, 31), (71, 129)]):
        path = tmp_path/f'{index}.jpg'
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    expected = load_and_preprocess_images(paths, image_resolution=64)
    actual = load_and_preprocess_images(paths, image_resolution=64, workers=4)
    assert torch.equal(expected, actual)


def test_attention_policy_only_selects_materially_faster_kernel_and_falls_back(monkeypatch):
    from types import SimpleNamespace
    from motius.pipelines.flowhmr.camera_attention import CameraAttention
    q = SimpleNamespace(is_cuda=True)
    k = SimpleNamespace(shape=(1, 16, 10000, 64))
    policy = CameraAttention()
    monkeypatch.setattr(policy, '_measure', lambda name,*args: .02 if name=='torch' else .01)
    calls = []
    def run(name,*args):
        calls.append((name, args))
        if name == 'cudnn': raise RuntimeError('unsupported tail shape')
        return 'reference-output'
    monkeypatch.setattr(policy, '_run', run)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    assert policy(q, k, k) == 'reference-output'
    assert policy.selected == 'torch'
    assert [name for name,args in calls] == ['cudnn', 'torch']
    assert all(args == (q,k,k) for name,args in calls)
    assert policy.fallback_reasons == ['unsupported tail shape']


def test_custom_attention_keeps_complete_keys_before_residual_mutation():
    torch.manual_seed(29)
    block = SelfAttentionBlock(64, 4, qkv_bias=True, init_values=.5, use_qk_norm=True).eval()
    x = torch.randn(2, 31, 64)
    key_counts = []
    def attention(q,k,v):
        key_counts.append(k.shape[-2])
        return torch.nn.functional.scaled_dot_product_attention(q,k,v)
    with torch.inference_mode():
        expected = block(x)
        actual = block.forward_chunked_(x.clone(), query_chunk_size=7, attention_fn=attention)
    assert key_counts == [31]*5
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_automatic_attention_handles_unavailable_cudnn_during_calibration(monkeypatch):
    from types import SimpleNamespace
    from motius.pipelines.flowhmr.camera_attention import CameraAttention
    policy = CameraAttention()
    def measure(name,*args):
        if name == 'cudnn': raise RuntimeError('No available cuDNN kernel')
        return .02
    monkeypatch.setattr(policy, '_measure', measure)
    monkeypatch.setattr(policy, '_run', lambda name,*args: name)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    assert policy(SimpleNamespace(is_cuda=True), SimpleNamespace(shape=(1, 16, 10000, 64)), None) == 'torch'
    assert policy.selected == 'torch'
    assert policy.fallback_reasons == ['No available cuDNN kernel']
