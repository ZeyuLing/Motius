"""Prepared GVHMR neural forwards synchronize gradients across two processes."""
import torch
from motius.models.gvhmr import GVHMRBundle


def _worker(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(7)
        bundle=GVHMRBundle(config=dict(latent_dim=32,num_layers=1,num_heads=4))
        bundle.model=DistributedDataParallel(bundle.model)
        optimizer=torch.optim.AdamW(bundle.model.parameters(),lr=1e-3)
        for step in range(2):
            torch.manual_seed(19+rank+step)
            optimizer.zero_grad()
            result=bundle.predict_motion(length=torch.tensor([10,8]),obs=torch.randn(2,10,17,3),
                f_cliffcam=torch.randn(2,10,3),f_cam_angvel=torch.randn(2,10,6),f_imgseq=torch.randn(2,10,1024))
            loss=sum(result[key].square().mean() for key in ['pred_x','pred_cam','static_conf_logits'])
            loss.backward();optimizer.step()
        value=bundle.network.embed_noisyobs.fc1.weight.detach()
        values=[torch.empty_like(value) for _ in range(2)];dist.all_gather(values,value)
        torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_process_network_training(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(f'file://{tmp_path / "rendezvous"}',),nprocs=2,join=True)
