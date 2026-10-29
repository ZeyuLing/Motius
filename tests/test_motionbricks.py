"""Native MotionBricks networks, artifact boundaries and real DDP gradients."""
from pathlib import Path
import json
import pytest
import torch
from motius.models.motionbricks import MotionBricksBundle
from motius.configuration.motionbricks import MotionBricksCheckpointLayout,validate_checkpoint_layout
from motius.motion.representation import get_spec


def options():
    vocabulary=dict(nb_code=16,code_dim=8,num_heads=2)
    pose=dict(min_tokens=2,max_tokens=4,down_t=2,pose_token_mlp_num_layers=2,root_feat_width=16,pose_feat_width=16,
        token_length_feat_width=8,text_emb_dim=16,text_embeddings=None,n_embd=32,n_head=4,n_layers=1,
        pose_vqvae=vocabulary,root_vqvae=vocabulary,local_root_dim=4,local_pose_dim=7)
    root=dict(min_tokens=2,max_tokens=4,down_t=2,use_hard_num_token_emb_for_root_prediction=True,
        pose_feat_dim=16,local_root_feat_dim=8,global_root_feat_dim=8,text_emb_dim=16,text_embeddings=None,
        input_feat_mlp_num_layers=2,n_embd=32,n_head=4,n_layers_shared=1,n_layers_root_token=1,
        width=16,depth=1,dilation_growth_rate=3,activation='relu',norm=None,
        local_root_dim=4,global_root_dim=5,local_pose_dim=7,pose_vqvae=vocabulary,root_vqvae=vocabulary)
    tokenizer=dict(encoder_state_dim=7,decoder_state_dim=10,decoder_target_cond_dim=7,decoder_external_cond_dim=2,
        nb_code=16,code_dim=8,num_heads=2,output_emb_width=8,down_t=2,stride_t=2,width=16,depth=1,kmeans_init=False)
    return dict(pose=pose,root=root,tokenizer=tokenizer)


def graphs(bundle):
    batch,frames,tokens=2,16,4
    lengths=torch.tensor([4,3])[:,None]
    pose=bundle.predict_pose(torch.randint(0,4,(batch,tokens,2)),torch.randn(batch,frames,4),torch.randn(batch,frames,7),
        torch.rand(batch,frames)>.5,lengths)['pose_logits']
    root=bundle.predict_root(torch.randn(batch,8,5),torch.ones(batch,8,dtype=torch.bool),torch.randn(batch,8,4),
        torch.ones(batch,8,dtype=torch.bool),torch.randn(batch,8,7),torch.ones(batch,8,dtype=torch.bool),
        torch.full((batch,1),6),chosen_token=lengths.reshape(-1)-2)
    codec=bundle.reconstruct(torch.randn(batch,frames,7),target_cond=torch.randn(batch,frames,7),
        has_target_cond=torch.rand(batch,frames)>.5,external_cond=torch.randn(batch,frames,2))
    return pose,root,codec


def test_motionbricks_representation_specs():
    assert get_spec('motionbricks_g1_414').dim==414
    assert get_spec('motionbricks_g1_413').dim==413
    assert get_spec('motionbricks_g1_418').dim==418
    assert get_spec('MotionBricks-G1-414').name=='motionbricks_g1_414'


def test_motionbricks_robot_geometry_is_not_in_models():
    import motius.configuration.motionbricks
    folder=Path(motius.configuration.motionbricks.__file__).parent/'motionbricks_assets/skeletons/g1'
    assert (folder/'g1.xml').is_file() and (folder/'scene_29dof.xml').is_file()


def test_missing_checkpoint_validation_does_not_construct_a_model(tmp_path):
    with pytest.raises(FileNotFoundError,match='missing:'):validate_checkpoint_layout(tmp_path)


def test_lfs_pointer_validation_does_not_construct_a_model(tmp_path):
    layout=MotionBricksCheckpointLayout.from_root(tmp_path)
    for path in [layout.clip,layout.pose,layout.root_model,layout.vqvae]:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(FileNotFoundError,match='git-lfs pointers'):validate_checkpoint_layout(tmp_path)


def test_motionbricks_checkpoint_roundtrip_saves_current_networks(tmp_path):
    torch.set_num_threads(1);model=MotionBricksBundle(options=options(),device='cpu')
    with torch.no_grad():next(model.pose.parameters()).add_(.5)
    model.save_pretrained(tmp_path)
    loaded=MotionBricksBundle.from_pretrained(tmp_path,device='cpu')
    assert all(torch.equal(value,loaded.state_dict()[key]) for key,value in model.state_dict().items())
    assert json.loads((tmp_path/'model_index.json').read_text())['pipeline_class']=='motius.pipelines.motionbricks.MotionBricksPipeline'
    torch.manual_seed(13);expected=graphs(model)
    torch.manual_seed(13);actual=graphs(loaded)
    assert torch.equal(expected[0],actual[0])
    assert torch.equal(expected[1]['pred_global_root_values'],actual[1]['pred_global_root_values'])
    assert torch.equal(expected[2]['recon_state'],actual[2]['recon_state'])


def _ddp(rank,path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(7);bundle=MotionBricksBundle(options=options(),device='cpu')
    for name in ['pose','root','encoder','decoder']:setattr(bundle,name,DDP(getattr(bundle,name),find_unused_parameters=True))
    bundle.train();optimizer=torch.optim.SGD(bundle.parameters(),lr=1e-4)
    for step in range(2):
        torch.manual_seed(101+step+rank);optimizer.zero_grad(set_to_none=True)
        pose,root,codec=graphs(bundle)
        loss=pose.square().mean()+root['num_token_logits'].square().mean()+root['pred_global_root_values'].square().mean()+codec['recon_state'].square().mean()
        loss.backward();optimizer.step()
    for name in ['pose','root','encoder','decoder']:
        value=next(getattr(bundle,name).parameters()).detach();gathered=[torch.empty_like(value) for _ in range(2)]
        dist.all_gather(gathered,value);assert torch.equal(gathered[0],gathered[1])
    dist.destroy_process_group()


def test_two_process_actual_neural_roots(tmp_path):
    torch.multiprocessing.spawn(_ddp,args=(str(tmp_path/'group'),),nprocs=2,join=True)


def test_motionbricks_synthetic_dataset_is_explicitly_a_dataset():
    from motius.datasets.motionbricks import SyntheticMotionDataset,collate_batch
    dataset=SyntheticMotionDataset(feat_dim=418,num_samples=2,min_frames=3,max_frames=4)
    batch=collate_batch([dataset[0],dataset[1]])
    assert batch['motion'].shape[0]==2 and batch['motion'].shape[-1]==418
    assert batch['motion_pad_mask'].dtype==torch.bool
