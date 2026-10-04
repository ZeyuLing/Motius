#!/usr/bin/env python3
"""Deterministic source parity on constructed motion inputs and real body assets."""
import sys, torch, json, copy, gc
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Compare full official FlowHMR losses, gradients and ODE sampling at zero tolerance.")
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--checkpoint-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation/flowhmr"))
options = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
options.output_dir.resolve().relative_to((ROOT / "outputs").resolve())
options.output_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(options.upstream))
from flowhmr.pipeline.pipeline_v2m import V2MPipeline
from motius.models.flowhmr import FlowHMRBundle
from motius.pipelines.flowhmr import FlowHMRPipeline
torch.set_num_threads(4)
torch.manual_seed(20261005)
root=options.checkpoint_root; out=options.output_dir
assets=dict(body_model_path=str(root/'body_models/smplh/neutral/model.npz'),j_regressor_path=str(root/'body_models/smpl_neutral_J_regressor.pt'))
reports={}
for variant in ['base','latest']:
 bundle=FlowHMRBundle.from_pretrained(root/f'flowhmr_{variant}',**assets,device='cuda')
 cfg=copy.deepcopy(bundle.config); args=cfg['train_pipeline_args']; args.update(smpl_model_path=assets['body_model_path'],j_regressor_path=assets['j_regressor_path'],mean_std=str(bundle.mean_std_path))
 official=V2MPipeline(network_module=cfg['network_module'],network_module_args=cfg['network_module_args'],**args).cuda()
 official.load_state_dict(bundle.model.state_dict(),strict=True)
 feature=torch.randn(1,4,3072,device='cuda'); camera=torch.eye(3,device='cuda').reshape(1,1,9).expand(1,4,9)
 entry={'body_sha256':__import__('hashlib').sha256(Path(assets['body_model_path']).read_bytes()).hexdigest(),'checkpoint_sha256':bundle.checkpoint_sha256}
 if variant=='base':
  rot=torch.tensor([1.,0.,0.,0.,1.,0.],device='cuda').expand(1,4,52,6).clone()
  shapes=torch.zeros(1,4,16,device='cuda'); trans=torch.zeros(1,4,3,device='cuda')
  with torch.no_grad():
   joints=bundle.model.body_model(dict(rot6d=rot.reshape(4,52,6),shapes=shapes.reshape(4,16),trans=trans.reshape(4,3)))['keypoints3d'].reshape(1,4,52,3)
  batch=dict(length=torch.tensor([4],device='cuda'),inputs={'feature':{'feature':feature,'camera_R':camera}},target=dict(smooth_root_vel=trans.clone(),smooth_root_pos=trans.clone(),local_joints_positions=joints,local_rot_data=rot,global_rot_data=rot.clone(),shapes=shapes,foot_contacts=torch.zeros(1,4,4,device='cuda')))
  entry['training']=[]
  for step in [1,10001]:
   bundle.train(); official.train(); official.global_iteration=step-1
   torch.manual_seed(73); a=official.forward_in_training(copy.deepcopy(batch))
   torch.manual_seed(73); b=bundle.training_forward(copy.deepcopy(batch),global_step=step)
   torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)
   for k in a['loss_dict']: torch.testing.assert_close(a['loss_dict'][k],b['loss_dict'][k],rtol=0,atol=0)
   a['loss'].backward(); b['loss'].backward(); count=0
   for (n,p),(m,q) in zip(official.named_parameters(),bundle.model.named_parameters()):
    assert n==m and (p.grad is None)==(q.grad is None)
    if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0); count+=1
   entry['training'].append({'step':step,'loss':a['loss'].item(),'exact_gradient_tensors':count,'atol':0,'rtol':0})
   official.zero_grad(set_to_none=True); bundle.zero_grad(set_to_none=True); del a,b
 bundle.eval(); official.eval()
 with torch.no_grad():
  a=official.generate({'feature':feature,'camera_R':camera},[13],4)
  b=bundle.generate_from_feature(feature,camera,seeds=[13])
 for k in a:
  if torch.is_tensor(a[k]): torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
 entry['sampling']={'frames':4,'seed':13,'exact_keys':[k for k in a if torch.is_tensor(a[k])],'atol':0,'rtol':0}
 reports[variant]=entry
 (out/'full_network_parity.json').write_text(json.dumps(reports,indent=2));print(variant,entry,flush=True)
 del official,bundle,a,b; gc.collect();torch.cuda.empty_cache()
