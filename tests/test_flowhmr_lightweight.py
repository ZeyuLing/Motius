"""Visibility, numerical and lifecycle contracts for lightweight inference."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import yaml

from motius.models.flowhmr.bundle import ASSETS
from motius.models.flowhmr.network.hymotion_mmdit_for_v2m import HunyuanMotionMMDiT
from motius.models.flowhmr.network.mmdit.query_attention import QueryAttentionMask, query_batched_attention


@pytest.mark.parametrize('chunk', [1, 3, 32])
def test_global_queries_keep_global_keys_and_padding_diagonal(chunk):
    torch.manual_seed(53)
    q, k, v = [torch.randn(2, 11, 2, 8) for _ in range(3)]
    valid = torch.tensor([[True] * 6 + [False] * 5, [False] * 11])
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed = torch.where(allowed.any(-1, keepdim=True), allowed, torch.eye(11, dtype=torch.bool)[None])
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed[:, None])
    actual = query_batched_attention(q, k, v, QueryAttentionMask(chunk, valid))
    torch.testing.assert_close(actual, expected.transpose(1, 2).reshape(2, 11, -1), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('mode', [None, 'causal', 'narrowband', 'narrowband_v2m'])
@pytest.mark.parametrize('tm', [0, -1, 2])
def test_complete_network_matches_dense_with_padding_and_unequal_streams(mode, tm):
    torch.manual_seed(17)
    cfg = yaml.safe_load((ASSETS / 'base_model.yml').read_text())['network_module_args']
    cfg.update(feat_dim=32, num_heads=2, num_layers=3, ctxt_input_dim={'feature': 8, 'camera_R': 9},
               vtxt_input_dim=8, mask_mode=mode, narrowband_length=.1,
               narrowband_v2m_length=2, narrowband_tm_length=tm, narrowband_tt_length=1,
               text_refiner_cfg={'num_layers': 1})
    network = HunyuanMotionMMDiT(**deepcopy(cfg)).eval()
    # Flow-matching heads and adaptive gates are zero-initialized. Activate them
    # so this comparison exercises attention rather than two constant zeros.
    with torch.no_grad():
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=.05)
                if module.bias is not None:
                    torch.nn.init.normal_(module.bias, std=.01)
    args = dict(x=torch.randn(2, 11, 491),
                ctxt_input={'feature': torch.randn(2, 7, 8), 'camera_R': torch.randn(2, 7, 9)},
                vtxt_input=torch.randn(2, 1, 8), timesteps=torch.tensor([.2, .7]),
                x_mask_temporal=torch.tensor([[True] * 8 + [False] * 3, [False] * 11]),
                ctxt_mask_temporal=torch.tensor([[True] * 3 + [False] * 4, [False] * 7]))
    with torch.no_grad():
        expected = network(**args)
        assert expected.abs().max() > 1e-6
        state = {key: value.clone() for key, value in network.state_dict().items()}
        network.set_attention_query_chunk_size(3)
        # The optimized forward must not allocate through either dense builder.
        network._build_dmm_attn_mask_shared = lambda **kw: pytest.fail('dense double mask allocated')
        network._build_smm_attn_mask_shared = lambda **kw: pytest.fail('dense single mask allocated')
        actual = network(**args)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert all(torch.equal(value, network.state_dict()[key]) for key, value in state.items())


def test_training_restores_original_attention_path():
    cfg = yaml.safe_load((ASSETS / 'base_model.yml').read_text())['network_module_args']
    cfg.update(feat_dim=32, num_heads=2, num_layers=3, text_refiner_cfg={'num_layers': 1})
    network = HunyuanMotionMMDiT(**deepcopy(cfg))
    reference = deepcopy(network)
    network.set_attention_query_chunk_size(5)
    network.train()
    assert network.training
    args = dict(x=torch.randn(1, 4, 491),
                ctxt_input={name: torch.randn(1, 4, size) for name, size in cfg['ctxt_input_dim'].items()},
                vtxt_input=torch.randn(1, 1, cfg['vtxt_input_dim']), timesteps=torch.tensor([.5]),
                x_mask_temporal=torch.ones(1, 4, dtype=torch.bool),
                ctxt_mask_temporal=torch.ones(1, 4, dtype=torch.bool))
    expected, actual = reference(**args), network(**args)
    assert torch.equal(actual, expected)
    expected.sum().backward(); actual.sum().backward()
    for original, changed in zip(reference.parameters(), network.parameters()):
        assert (original.grad is None) == (changed.grad is None)
        if original.grad is not None:
            assert torch.equal(original.grad, changed.grad)
    # Runtime configuration is not a learned state or an exported weight.
    assert not any('query_chunk' in key for key in network.state_dict())
    for value in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            network.set_attention_query_chunk_size(value)
    network.set_attention_query_chunk_size(0)
    assert network.attention_query_chunk_size == 0


def test_token_only_keeps_first_body_and_both_original_hands():
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor

    class Decoder(torch.nn.Module):
        def forward(self, value):
            return value, {}

    class Estimator:
        def __init__(self):
            self.faces = np.zeros((0, 3), dtype=np.int64)
            self.model = SimpleNamespace(decoder=Decoder(), decoder_hand=Decoder())
            self.tail_calls = 0

        def process_one_image(self, *args, **kwargs):
            self.model.decoder(torch.full((1, 3, 1024), 1.))
            self.model.decoder_hand(torch.full((1, 3, 1024), 2.))
            self.model.decoder_hand(torch.full((1, 3, 1024), 3.))
            self.tail_calls += 1
            self.model.decoder(torch.full((1, 3, 1024), 9.))
            return [{}]

    reference, optimized = Estimator(), Estimator()
    full = Sam3DTokenExtractor(reference)
    light = Sam3DTokenExtractor(optimized, token_only=True)
    for _ in range(2):
        expected, _ = full.extract_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        actual, outputs = light.extract_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        assert torch.equal(actual, expected)
        assert torch.equal(actual.reshape(3, 1024)[:, 0], torch.tensor([1., 2., 3.]))
        assert outputs == []
    assert reference.tail_calls == 2 and optimized.tail_calls == 0
    full.close(); light.close()
    assert not optimized.model.decoder_hand._forward_hooks


def test_frontend_failure_releases_models_before_motion_restore(monkeypatch):
    from motius.pipelines.flowhmr import lightweight
    events = []

    class Module:
        def __init__(self, name): self.name = name
        def to(self, device): events.append((self.name, str(device))); return self

    monkeypatch.setattr(lightweight, 'release_allocator', lambda device: None)
    pipeline = SimpleNamespace(lightweight=True, bundle=Module('bundle'), codec=Module('codec'))
    pre = SimpleNamespace(release=lambda: events.append(('pre', 'release')))
    with pytest.raises(RuntimeError, match='frontend failed'):
        with lightweight.frontend_resources(pipeline, pre, 'cuda:2'):
            raise RuntimeError('frontend failed')
    assert events == [('bundle', 'cpu'), ('codec', 'cpu'), ('pre', 'release'),
                      ('bundle', 'cuda:2'), ('codec', 'cuda:2')]


def test_body_model_resolution_does_not_substitute_gender(tmp_path, monkeypatch):
    from motius.pipelines.flowhmr.lightweight import resolve_body_model
    monkeypatch.delenv('MOTIUS_SMPLH_MODEL', raising=False)
    monkeypatch.setenv('MOTIUS_BODY_MODEL_DIR', str(tmp_path))
    neutral = tmp_path / 'smplh/neutral/model.npz'
    neutral.parent.mkdir(parents=True)
    neutral.write_bytes(b'fixture')
    assert resolve_body_model() == str(neutral)
    with pytest.raises(FileNotFoundError):
        resolve_body_model(tmp_path / 'missing.npz')


@pytest.mark.parametrize('sources,expected_checkpoint', [
    (['input.mp4'], 'ZeyuLing/Motius-FlowHMR-Latest'),
    (['local_checkpoint', 'input.mp4'], 'local_checkpoint'),
])
def test_one_command_cli_and_legacy_checkpoint_argument(tmp_path, monkeypatch, sources, expected_checkpoint):
    import sys
    from tools import infer_flowhmr as entry
    body = tmp_path / 'model.npz'; body.write_bytes(b'fixture')
    captured = {}

    def load(checkpoint, **kwargs):
        captured.update(checkpoint=checkpoint, loader_kwargs=kwargs)
        def infer(video, **kwargs):
            captured.update(video=video, inference_kwargs=kwargs)
            return object()
        return SimpleNamespace(infer_monocular_motion_capture=infer)

    monkeypatch.setattr(entry.FlowHMRPipeline, 'from_pretrained', load)
    monkeypatch.setattr(entry, 'save_monocular_capture_result', lambda result, path: captured.update(output=path))
    monkeypatch.setattr(sys, 'argv', ['infer_flowhmr.py', *sources, '--lightweight', '--body-model', str(body), '--work-dir', str(tmp_path)])
    entry.main()
    assert captured['checkpoint'] == expected_checkpoint
    assert captured['video'] == 'input.mp4'
    assert captured['loader_kwargs']['lightweight'] is True
    assert captured['inference_kwargs']['vggt_frame_interval'] == 1
    assert captured['output'] == tmp_path / 'result.npz'


def test_conditioning_path_skips_post_decoder_work_and_removes_temporary_hooks(monkeypatch):
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam import inference

    class Decoder(torch.nn.Module):
        def forward(self, value):
            return value, {}

    model=SimpleNamespace(decoder=Decoder(),decoder_hand=Decoder())
    hands=[]
    def forward_step(model,batch,decoder_type):
        if decoder_type=='body':
            model.decoder(torch.full((1,1,1024),1.))
            return {}
        hands.append(batch)
        model.decoder_hand(torch.full((1,1,1024),float(len(hands)+1)))
        pytest.fail('Post-decoder hand processing is unused by FlowHMR conditioning')
    monkeypatch.setattr(inference,'_forward_step',forward_step)
    left=np.array([[1.,2.,4.,6.]])
    right=np.array([[5.,2.,7.,6.]])
    monkeypatch.setattr(inference,'_get_hand_box',lambda *args:(left.copy(),right.copy()))
    def prepare(image,transform,boxes,**kwargs):
        return {'image':image,'boxes':boxes,'img':torch.zeros(1,1,3,8,8)}
    monkeypatch.setattr(inference,'prepare_batch',prepare)
    class Estimator:
        faces=np.zeros((0,3),dtype=np.int64)
        def __init__(self):self.model=model
        def process_one_image(self,image,**kwargs):
            inference.run_conditioning_inference(model,image,
                {'img':torch.zeros(1,1,3,8,8),'cam_int':torch.eye(3)[None]},None)
            pytest.fail('The outer capture hook must finish after the right-hand token')
    extractor=Sam3DTokenExtractor(Estimator(),token_only=True)
    image=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
    token,_=extractor.extract_frame(image)
    assert torch.equal(token.reshape(3,1024)[:,0],torch.tensor([1.,2.,3.]))
    np.testing.assert_array_equal(hands[0]['image'],image[:,::-1])
    np.testing.assert_array_equal(hands[0]['boxes'],np.array([[3.,2.,6.,6.]]))
    np.testing.assert_array_equal(hands[1]['boxes'],right)
    assert len(model.decoder_hand._forward_hooks)==1
    extractor.close()
    assert not model.decoder_hand._forward_hooks


@pytest.mark.parametrize('padding', [0.9, 1.25])
def test_gpu_preprocessing_matches_reference_pixels_and_geometry(padding):
    from torchvision.transforms import ToTensor
    from motius.pipelines.flowhmr.sam_data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from motius.pipelines.flowhmr.sam_data.utils.prepare_batch import prepare_batch
    from motius.pipelines.flowhmr.sam.gpu_preprocess import prepare_gpu_batch, merge_hands
    rng=np.random.default_rng(37)
    image=rng.integers(0,256,(73,91,3),dtype=np.uint8)
    # Fractional, outside-image and tiny boxes exercise quantization/border rules.
    boxes=np.array([[4.3,7.8,71.2,65.1],[-11.9,-8.1,16.8,21.2],[37.15,22.33,39.3,24.98]],np.float32)
    transform=Compose([GetBBoxCenterScale(padding=padding),TopdownAffine((48,64),use_udp=False),VisionTransformWrapper(ToTensor())])
    camera=torch.tensor([[[110.,0.,45.5],[0.,110.,36.5],[0.,0.,1.]]])
    reference=prepare_batch(image,transform,boxes,cam_int=camera)
    actual=prepare_gpu_batch(torch.from_numpy(image),boxes,camera,transform)
    for key in ('img','bbox','bbox_center','bbox_scale','affine_trans','img_size','ori_img_size','mask','mask_score','person_valid','cam_int'):
        torch.testing.assert_close(reference[key],actual[key],rtol=0,atol=0)
    merged=merge_hands(actual,actual)
    assert merged['img'].shape[:2]==(1,6)
    assert torch.equal(merged['img'][:,:3],actual['img'])
    assert torch.equal(merged['cam_int'],camera)


def test_fast_capture_splits_hand_batch_in_left_right_order():
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor, _TokensCaptured
    class Decoder(torch.nn.Module):
        def forward(self,value):return value,{}
    model=SimpleNamespace(decoder=Decoder(),decoder_hand=Decoder())
    extractor=Sam3DTokenExtractor(SimpleNamespace(model=model,faces=np.zeros((0,3))),token_only=True,fast=True)
    extractor._capture_active=True
    model.decoder(torch.full((1,3,1024),1.))
    with pytest.raises(_TokensCaptured):
        model.decoder_hand(torch.cat([torch.full((1,3,1024),2.),torch.full((1,3,1024),3.)]))
    assert len(extractor._hand_tokens)==2
    assert torch.equal(extractor._hand_tokens[0],torch.full((1,3,1024),2.))
    assert torch.equal(extractor._hand_tokens[1],torch.full((1,3,1024),3.))
    extractor.close()


def test_fast_frontend_runs_one_hand_call_and_restores_batch_state(monkeypatch):
    from torchvision.transforms import ToTensor
    from motius.pipelines.flowhmr.sam_data.transforms import Compose,GetBBoxCenterScale,TopdownAffine,VisionTransformWrapper
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam import inference
    class Decoder(torch.nn.Module):
        def forward(self,value):return value,{}
    class Model:
        decoder=Decoder();decoder_hand=Decoder()
        def _initialize_batch(self,batch):self.persons=batch['img'].shape[1]
    model=Model();calls=[]
    def forward(model,batch,decoder_type):
        calls.append((decoder_type,model.persons))
        if decoder_type=='body':
            model.decoder(torch.full((1,1,1024),1.));return {}
        assert model.persons==2
        model.decoder_hand(torch.cat((torch.full((1,1,1024),2.),torch.full((1,1,1024),3.))))
    monkeypatch.setattr(inference,'_forward_step',forward)
    monkeypatch.setattr(inference,'_get_hand_box',lambda *args:(np.array([[1.,1.,4.,5.]],np.float32),np.array([[4.,1.,7.,5.]],np.float32)))
    transform=Compose([GetBBoxCenterScale(),TopdownAffine((8,8),use_udp=False),VisionTransformWrapper(ToTensor())])
    estimator=SimpleNamespace(model=model,device=torch.device('cpu'),faces=np.zeros((0,3)),transform=transform,transform_hand=transform)
    extractor=Sam3DTokenExtractor(estimator,token_only=True,fast=True)
    features,outputs=extractor.extract_frame(np.zeros((8,8,3),np.uint8),bboxes=np.array([[0.,0.,8.,8.]],np.float32),cam_int=torch.eye(3)[None])
    assert calls==[('body',1),('hand',2)]
    assert model.persons==1
    assert outputs==[]
    assert torch.equal(features.reshape(3,1024)[:,0],torch.tensor([1.,2.,3.]))
    extractor.close()


@pytest.mark.parametrize('mean,std', [([.485,.456,.406],[.229,.224,.225]),([123.675,116.28,103.53],[58.395,57.12,57.375])])
def test_known_unit_rgb_normalization_matches_reference(mean,std):
    from motius.motion.conditioning.sam3d_images import prepare_sam3d_images
    cfg=SimpleNamespace(MODEL=SimpleNamespace(IMAGE_MEAN=mean,BACKBONE=SimpleNamespace(TYPE='dinov3_vitl16')))
    model=SimpleNamespace(cfg=cfg,image_mean=torch.tensor(mean)[:,None,None],image_std=torch.tensor(std)[:,None,None])
    pixels=torch.randint(0,256,(1,2,3,8,8)).float()/255
    a=prepare_sam3d_images(model,{'img':pixels.clone()})['network_images']
    b=prepare_sam3d_images(model,{'img':pixels.clone()},unit_rgb=True)['network_images']
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_fast_frame_capture_keeps_each_frames_body_and_both_hands(monkeypatch):
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam import inference
    class Decoder(torch.nn.Module):
        def forward(self, value): return value, {}
    model = SimpleNamespace(decoder=Decoder(), decoder_hand=Decoder())
    estimator = SimpleNamespace(model=model, faces=np.zeros((0, 3), dtype=np.int64))
    extractor = Sam3DTokenExtractor(estimator, token_only=True, fast=True)
    def batched(estimator, images, boxes, cameras, *, cache):
        values = torch.arange(3.).reshape(3, 1, 1).expand(3, 1, 1024)
        model.decoder(values)
        model.decoder_hand(torch.cat([values+10, values+20]))
    monkeypatch.setattr(inference, 'run_fast_frame_batch', batched)
    try:
        result = extractor.extract_frames([None]*3, [None]*3, [None]*3)
        assert result.shape == (3, 3072)
        torch.testing.assert_close(result[:, [0, 1024, 2048]], torch.tensor([[0, 10, 20], [1, 11, 21], [2, 12, 22.]]))
        assert extractor._capture_active is False
    finally: extractor.close()


def test_fast_frame_batch_chunks_paired_hands_without_losing_order(monkeypatch):
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam import inference, gpu_preprocess
    class Decoder(torch.nn.Module):
        def forward(self, value): return value, {}
    class Model:
        def __init__(self):
            self.decoder = Decoder()
            self.decoder_hand = Decoder()
        def _initialize_batch(self, batch): pass
    model = Model()
    calls = []
    prepared = []
    def prepare(image, box, camera, transform):
        i = len(prepared)
        value = i if i < 6 else (10 if (i-6) % 2 == 0 else 20) + (i-6)//2
        prepared.append(value)
        return {'img': torch.tensor([[[float(value)]]])}
    def forward(model, batch, decoder_type):
        values = batch['img'].reshape(-1, 1, 1).expand(-1, 1, 1024)
        calls.append((decoder_type, len(values)))
        if decoder_type == 'body':
            model.decoder(values)
            return {}
        model.decoder_hand(values)
    monkeypatch.setattr(gpu_preprocess, 'prepare_gpu_batch', prepare)
    monkeypatch.setattr(inference, '_forward_step', forward)
    monkeypatch.setattr(inference, '_get_hand_box', lambda *args: (np.zeros((6,4)), np.zeros((6,4))))
    estimator = SimpleNamespace(model=model, device=torch.device('cpu'), faces=np.zeros((0,3)),
                                transform=None, transform_hand=None)
    extractor = Sam3DTokenExtractor(estimator, token_only=True, fast=True)
    try:
        result = extractor.extract_frames([np.zeros((8,8,3), np.uint8)]*6, [np.zeros(4)]*6, [None]*6)
        assert calls == [('body', 6), ('hand', 8), ('hand', 4)]
        expected = torch.tensor([[i, i+10, i+20] for i in range(6)], dtype=torch.float32)
        torch.testing.assert_close(result[:, [0, 1024, 2048]], expected)
    finally:
        extractor.close()


def test_video_frame_batches_keep_tail_cameras_and_live_counts(monkeypatch):
    from motius.pipelines.flowhmr import features
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(5)]
    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return True
        def get(self, code): return len(frames)
        def read(self):
            if self.index == len(frames): return False, None
            frame = frames[self.index]; self.index += 1
            return True, frame
        def release(self): pass
    monkeypatch.setattr(features.cv2, 'VideoCapture', Capture)
    extractor = features.Sam3DTokenExtractor.__new__(features.Sam3DTokenExtractor)
    extractor.fast = True
    extractor.token_only = True
    calls = []
    def extract(images, boxes, cameras):
        ids = [int(image[0, 0, 0]) for image in images]
        np.testing.assert_array_equal(boxes[:, 0], ids)
        np.testing.assert_array_equal(cameras[:, 0, 0], ids)
        calls.append(ids)
        return torch.tensor(ids, dtype=torch.float32)[:, None].expand(-1, 3072)
    extractor.extract_frames = extract
    progress = []
    value = extractor.extract_video_tokens('fixture', bbox_xyxy=torch.arange(5)[:, None].expand(5, 4),
        K_all=torch.arange(5)[:, None, None].expand(5, 3, 3), frame_batch_size=2,
        progress_cb=lambda stage,current,total,message: progress.append((current,total)))
    assert calls == [[0, 1], [2, 3], [4]]
    torch.testing.assert_close(value[:, 0], torch.arange(5, dtype=torch.float32))
    assert progress[0] == (0, 5) and progress[-1] == (5, 5)
    assert (2, 5) in progress and (4, 5) in progress


def test_conditioning_decoder_omits_only_unused_final_prediction():
    from motius.models.vision.sam3d_body.decoders.promptable_decoder import PromptableDecoder
    torch.manual_seed(71)
    decoder = PromptableDecoder(dims=16, context_dims=16, depth=3, num_heads=2,
        head_dims=8, mlp_dims=32, layer_scale_init_value=0.,
        do_interm_preds=True, keypoint_token_update=True)
    decoder.eval()
    tokens, image = torch.randn(2, 4, 16), torch.randn(2, 16, 3, 3)
    calls, updates = [], []
    def predict(value, prev_pose_output, layer_idx):
        calls.append(layer_idx)
        return {'geometry': value.sin()}
    def update(value, augment, pose, layer_idx):
        updates.append(layer_idx)
        return value+pose['geometry']*.03, augment, None, None
    with torch.no_grad():
        reference, _ = decoder(tokens, image, token_to_pose_output_fn=predict,
                              keypoint_token_update_fn=update)
        assert calls == [0, 1, 2] and updates == [0, 1]
        calls.clear(); updates.clear()
        actual, predictions = decoder(tokens, image, token_to_pose_output_fn=predict,
            keypoint_token_update_fn=update, predict_final=False)
    assert calls == [0, 1] and updates == [0, 1]
    assert predictions[-1] == {}
    assert torch.equal(reference, actual)


def test_conditioning_ray_cache_preserves_rays_with_changing_camera_and_boxes():
    from motius.models.vision.sam3d_body.meta_arch.sam3d_body import SAM3DBody
    owner = SimpleNamespace(_conditioning_constant=SAM3DBody._conditioning_constant)
    torch.manual_seed(15)
    batch = {'img': torch.randn(3, 1, 3, 8, 8),
             'affine_trans': torch.randn(3, 1, 2, 3),
             'cam_int': torch.randn(3, 3, 3)}
    batch['affine_trans'][..., [0, 1], [0, 1]] += 3
    batch['cam_int'][..., [0, 1], [0, 1]] += 5
    cache = {}
    for _ in range(2):
        expected = SAM3DBody.get_ray_condition(owner, batch)
        actual = SAM3DBody.get_ray_condition(owner,
            dict(batch, _conditioning_only=True, _conditioning_cache=cache))
        assert torch.equal(expected, actual)
        batch['cam_int'] += .2
        batch['affine_trans'] += .1
    assert len(cache) == 1


def test_official_frame_batch_retains_cpu_crops_and_separate_hands(monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam import inference

    class Decoder(torch.nn.Module):
        def forward(self, tokens):
            return tokens, {}
    class Model:
        def __init__(self):
            self.decoder, self.decoder_hand = Decoder(), Decoder()
            self.calls=[]
        def _initialize_batch(self,batch):
            self.size=batch['img'].shape[0]
    model=Model()
    estimator=SimpleNamespace(model=model,faces=None,device=torch.device('cpu'),transform='body',transform_hand='hand')
    prepared=[]
    def prepare(image,transform,boxes,cam_int=None,**kwargs):
        prepared.append((transform,np.array(image),np.array(boxes),cam_int.clone()))
        marker=float(image[0,0,0])
        return {'img':torch.full((1,1,3,2,2),marker),'cam_int':cam_int.float(),'img_ori':[None]}
    def forward(model,batch,decoder_type):
        model.calls.append((decoder_type,batch['img'].shape[0],batch['cam_int'][:,0,0].tolist()))
        tokens=batch['img'][:,0,0,0,0,None,None]
        return (model.decoder if decoder_type=='body' else model.decoder_hand)(tokens)
    monkeypatch.setattr(inference,'prepare_batch',prepare)
    monkeypatch.setattr(inference,'_forward_step',forward)
    monkeypatch.setattr(inference,'_get_hand_box',lambda model,body,batch:(np.array([[0,0,1,1],[0,0,1,1]],dtype=float),np.array([[1,0,2,1],[1,0,2,1]],dtype=float)))
    images=[np.repeat(np.arange(3,dtype=np.uint8)[None,:,None],3,2),np.repeat((np.arange(3,dtype=np.uint8)+10)[None,:,None],3,2)]
    cameras=[torch.eye(3),torch.eye(3)*2]
    extractor=Sam3DTokenExtractor(estimator,token_only=True,fast=False)
    try:
        result=extractor.extract_frames(images,[[0,0,2,1]]*2,cameras)
    finally:extractor.close()
    assert result.tolist()==[[0.,2.,0.],[10.,12.,10.]]
    assert model.calls==[('body',2,[1.,2.]),('hand',2,[1.,2.]),('hand',2,[1.,2.])]
    assert [p[0] for p in prepared]==['body','body','hand','hand','hand','hand']


def test_sam_float32_mode_upcasts_checkpoint_values_without_inventing_precision(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from motius.models.vision.sam3d_body import bundle as sam_bundle
    class Config:
        def __init__(self):
            self.MODEL=SimpleNamespace(MHR_HEAD=SimpleNamespace(MHR_MODEL_PATH=''),BACKBONE=SimpleNamespace(TYPE='test'))
            self.TRAIN=SimpleNamespace(USE_FP16=True, FP16_TYPE='bfloat16')
        def defrost(self):pass
        def freeze(self):pass
    class Dummy(torch.nn.Module):
        def __init__(self,cfg,init_backbone_device=None):
            super().__init__()
            dtype=(torch.float16 if cfg.TRAIN.FP16_TYPE=='float16' else torch.bfloat16) if cfg.TRAIN.USE_FP16 else torch.float32
            self.backbone=torch.nn.Linear(2,1,bias=False,dtype=dtype)
    monkeypatch.setattr(sam_bundle,'get_config',lambda path:Config())
    monkeypatch.setattr(sam_bundle,'SAM3DBody',Dummy)
    monkeypatch.setattr(sam_bundle,'load_state_dict',lambda model,state,strict:model.load_state_dict(state,strict=strict))
    checkpoint=tmp_path/'model.ckpt'
    stored=torch.tensor([[1.125,-2.75]],dtype=torch.bfloat16)
    torch.save({'state_dict':{'backbone.weight':stored}},checkpoint)
    default=sam_bundle.SAM3DBodyBundle(checkpoint,'mhr.pt',config_path='config.yaml',precision='source')
    comparison=sam_bundle.SAM3DBodyBundle(checkpoint,'mhr.pt',config_path='config.yaml',precision='float32')
    half=sam_bundle.SAM3DBodyBundle(checkpoint,'mhr.pt',config_path='config.yaml',precision='float16')
    assert default.model.backbone.weight.dtype==torch.bfloat16
    assert comparison.model.backbone.weight.dtype==torch.float32
    assert half.model.backbone.weight.dtype==torch.float16
    assert comparison.model.checkpoint_backbone_dtypes==['torch.bfloat16']
    torch.testing.assert_close(comparison.model.backbone.weight,stored.float(),rtol=0,atol=0)
    torch.testing.assert_close(half.model.backbone.weight,stored.half(),rtol=0,atol=0)
    torch.save({'state_dict':{'backbone.weight':torch.tensor([[70000.,0.]])}},checkpoint)
    with pytest.raises(ValueError, match='cannot be represented in FP16'):
        sam_bundle.SAM3DBodyBundle(checkpoint,'mhr.pt',config_path='config.yaml',precision='float16')


def test_strict_float32_scope_restores_backend_settings_on_error():
    import torch
    from types import SimpleNamespace
    from motius.pipelines.flowhmr.pipeline import _strict_frontend_float32
    seen=[]
    @_strict_frontend_float32
    def fail(self):
        seen.append((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
        raise ValueError('expected')
    original=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        with pytest.raises(ValueError,match='expected'):fail(SimpleNamespace(frontend_precision='float32'))
        assert seen==[(False,False)]
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=original
