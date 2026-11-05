"""Compare complete native MotionBricks navigation with the independent original components."""
from __future__ import annotations
import argparse,ast,json,os,subprocess,sys
from pathlib import Path
from verify_bailando_source_parity import compare
from verify_intergen_source_parity import digest


def reference(args):
    import torch,numpy as np,copy,mujoco
    from scipy.spatial.transform import Rotation as R
    from omegaconf import OmegaConf
    sys.path.insert(0,str(args.upstream/'motionbricks'))
    from motionbricks.helper.pl_util import load_motion_rep
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
    from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE
    from motionbricks.motion_backbone.models.pose_model import MotionModel as PoseModel
    from motionbricks.motion_backbone.models.root_model import MotionModel as RootModel
    from motionbricks.motion_backbone.inference.motion_inference import motion_inference
    from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
    from motionbricks.motion_backbone.demo.clips import clip_holder_G1
    root=args.artifact/'motionbricks_checkpoint'
    cfg=OmegaConf.load(root/'motionbricks_pose/version_1/hparams.yaml')
    cfg.skeleton.folder=str(root/'motionbricks_pose/version_1/skeleton');cfg.motion_rep.stats.folder=str(root/'motionbricks_pose/version_1/stats/motion')
    rep=load_motion_rep(cfg);pose=pose_backbone_network(rep,cfg.model.backbone_network.args)
    root_cfg=OmegaConf.load(root/'motionbricks_root/version_1/hparams.yaml')
    root_network=root_backbone_network(root_cfg.model.backbone_network.args,rep)
    options=OmegaConf.to_container(cfg.model.pose_vqvae_network,resolve=True);options.pop('_target_');options.pop('motion_rep')
    vq=VQVAE(motion_rep=rep.dual_rep.local_motion_rep,**options)
    cfg.model.args.vqvae_model_ckpt_path=str(root/'motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt')
    models=dict(pose=PoseModel(vq,None,pose,rep,args=cfg.model.args),root=RootModel(None,None,root_network,rep,args=root_cfg.model.args))
    for name,model in models.items():
        state=torch.load(root/('motionbricks_'+name)/'version_1/checkpoints/model-step=2000000.ckpt',map_location='cpu',weights_only=False,mmap=True)['state_dict']
        model.load_state_dict(state,strict=True)
    torch.set_float32_matmul_precision(root_cfg.get('matmul_precision','highest'))
    # Compile the original headless controller classes unchanged; importing its
    # unrelated keyboard listener would require a desktop display in a GPU pod.
    file=args.upstream/'motionbricks/motionbricks/motion_backbone/demo/controllers.py'
    tree=ast.parse(file.read_text());scope=dict(t=torch,np=np,mujoco=mujoco,R=R,copy=copy,clip_holder_G1=clip_holder_G1)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'base_controller','random_controller'}],type_ignores=[]),str(file),'exec'),scope)
    return models,motion_inference,full_navigation_agent,scope['random_controller']


def worker(args):
    import random,numpy as np,torch
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    counters=dict(pose=0,root=0)
    if args.worker=='upstream':
        models,Inference,Agent,Controller=reference(args)
        networks={name:models[name].backbone_net for name in ['pose','root']}
    else:
        from motius.models.motionbricks import MotionBricksBundle
        from motius import Pipeline
        from motius.pipelines.motionbricks.pipeline import MotionBricksPipeline
        pipeline=Pipeline.from_pretrained(str(args.artifact),controller='random',bundle_kwargs={'device':'cuda'})
        bundle=pipeline.bundle
        networks={name:getattr(bundle,name) for name in ['pose','root']}
    for name,network in networks.items():
        def hook(module,inputs,outputs,key=name):counters[key]+=1
        network.register_forward_hook(hook)
    outputs={}
    for case,seed in enumerate([1234,1241]):
        if args.worker=='upstream':
            from types import SimpleNamespace
            import mujoco
            inference=Inference(models,models['pose'].args,device='cuda')
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            assets=args.upstream/'motionbricks/assets/skeletons/g1'
            with torch.inference_mode():
                agent=Agent(inference,None,device='cuda',speed_scale=[1.,1.],target_root_realignment=True,source_root_realignment=True,
                    force_canonicalization=True,skeleton_xml=str(assets/'g1.xml'),skip_ending_target_cond=False,filter_qpos=True,
                    clips='G1',ckpt_path=str(args.artifact/'motionbricks_checkpoint/G1-clip.ckpt'),reprocess_clips=False).cuda()
                control=Controller(disable_running=True,lookat_movement_direction=False,new_control_dt=2.,max_angle_change_between_controls=.5*np.pi,
                    clips='G1',min_token=inference._args['min_tokens'],max_token=inference._args['max_tokens'])
                model=mujoco.MjModel.from_xml_path(str(assets/'scene_29dof.xml'));data=mujoco.MjData(model);model.opt.timestep=1/30
                agent.reset();qpos=[];steps=160;tail=min(100,steps//5)
                for step in range(steps):
                    frame=agent.get_next_frame();qpos.append(np.asarray(frame,dtype=np.float32).copy())
                    context=agent.get_context_mujoco_qpos();data.qpos[:]=frame
                    signals=control.generate_control_signals(None,model,data,visualize=False,control_info=dict(force_idle=step>=steps-tail,allowed_mode=None))
                    signals['context_mujoco_qpos']=context
                    agent.generate_new_frames(signals,control.get_controller_dt()*2.)
                result=dict(qpos=np.stack(qpos),fps=30)
        else:
            with torch.inference_mode():
                result=pipeline.infer_g1_qpos_generation(steps=160,seed=seed)
        outputs[str(case)]={'qpos':torch.from_numpy(result['qpos'])}
        assert result['fps']==30
    assert counters['pose']>0 and counters['root']>0,counters
    torch.save(outputs,args.output_dir/(args.worker+'.pt'))
    (args.output_dir/(args.worker+'-calls.json')).write_text(json.dumps(counters))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'rollout_parity.json';report=dict(complete=False,comparison='Original MotionBricks real G1 primitives/controller, root and pose sampling, motion codec, canonicalization and MuJoCo qpos assembly versus native composition',seeds=[1234,1241],frames_per_case=160,controller='random',reference_boundary='Original headless controller classes extracted unchanged to omit unused desktop keyboard imports; seed applied after NN construction in both workers',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream/'motionbricks' if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__,network_calls={kind:json.loads((args.output_dir/(kind+'-calls.json')).read_text()) for kind in ['upstream','native']})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All MotionBricks full rollout comparisons exact')

if __name__=='__main__':main()
