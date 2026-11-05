"""Compare GVHMR network, camera decoding, contact correction and IK to source."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from omegaconf import OmegaConf
        from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import Pipeline
        from hmr4d.utils.geo.hmr_cam import normalize_kp2d
        config=OmegaConf.create(dict(endecoder_opt={'_target_':'hmr4d.model.gvhmr.utils.endecoder.EnDecoder','stats_name':'MM_V1_AMASS_LOCAL_BEDLAM_CAM'},normalize_cam_angvel=True,weights=None,static_conf=None))
        prediction=Pipeline(config,OmegaConf.create({'_target_':'hmr4d.network.gvhmr.relative_transformer.NetworkEncoderRoPE'})).to(args.device).eval()
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['state_dict']
        prediction.load_state_dict({k.removeprefix('pipeline.'):v for k,v in state.items()},strict=True)
    else:
        from motius.models.gvhmr import GVHMRBundle
        from motius.pipelines.gvhmr.prediction import GVHMRPrediction
        from motius.motion.representation.gvhmr.camera import normalize_kp2d
        bundle=GVHMRBundle(checkpoint_path=args.checkpoint,device=args.device)
        prediction=GVHMRPrediction(bundle,args.body_model)
    output={}
    for index,(batch,frames,static,postproc) in enumerate([(1,30,False,False),(1,30,False,True),(2,50,False,False),(1,50,True,True)]):
        torch.manual_seed(31+index)
        length=torch.tensor([frames,frames-7][:batch],device=args.device)
        bbox=torch.tensor([320.,240.,400.],device=args.device).expand(batch,frames,3).clone()
        intrinsic=torch.tensor([[800.,0.,320.],[0.,800.,240.],[0.,0.,1.]],device=args.device).expand(batch,frames,3,3).clone()
        keypoints=torch.rand(batch,frames,17,3,device=args.device);keypoints[...,:2]=keypoints[...,:2]*200+200
        from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_rotation_6d
        angular=matrix_to_rotation_6d(axis_angle_to_matrix(torch.randn(batch,frames,3,device=args.device)*.01))
        data=dict(length=length,obs=normalize_kp2d(keypoints,bbox),bbx_xys=bbox,K_fullimg=intrinsic,
                  cam_angvel=angular,f_imgseq=torch.randn(batch,frames,1024,device=args.device)*.1)
        with torch.no_grad():
            result=prediction.forward(data,postproc=postproc,static_cam=static)
        def cpu(value):
            if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
            return value.detach().cpu() if value is not None else None
        output[f'batch{batch}_frames{frames}_static{static}_postproc{postproc}']=cpu(result)
    # Released complete training objective, with the same physical labels and masks.
    torch.manual_seed(89)
    batch,frames=2,16
    pose=torch.randn(batch,frames,63,device=args.device)*.05
    shape=torch.randn(batch,frames,10,device=args.device)*.1
    orientation=torch.randn(batch,frames,3,device=args.device)*.05
    translation=torch.randn(batch,frames,3,device=args.device)*.02
    translation[...,2]+=3.
    labels=dict(body_pose=pose,betas=shape,global_orient=orientation,transl=translation)
    codec=prediction.endecoder
    with torch.no_grad():
        vertices,joints17=codec.smplx_model(**labels)
        relative_vertices=vertices-joints17[:,:,[11,12]].mean(-2,keepdim=True)
        joints=codec.fk_v2(**labels)
    lens=torch.tensor([16,12],device=args.device)
    mask=torch.arange(frames,device=args.device)[None,:]<lens[:,None]
    bbox=torch.tensor([320.,240.,400.],device=args.device).expand(batch,frames,3).clone()
    intrinsic=torch.tensor([[800.,0.,320.],[0.,800.,240.],[0.,0.,1.]],device=args.device).expand(batch,frames,3,3).clone()
    data=dict(length=lens,obs=torch.randn(batch,frames,17,3,device=args.device),bbx_xys=bbox,K_fullimg=intrinsic,
        cam_angvel=matrix_to_rotation_6d(torch.eye(3,device=args.device).expand(batch,frames,3,3)),
        f_imgseq=torch.randn(batch,frames,1024,device=args.device)*.1,smpl_params_c=labels,
        smpl_params_w={k:v.clone() for k,v in labels.items()},R_c2gv=torch.eye(3,device=args.device).expand(batch,frames,3,3),
        mask=dict(valid=mask,spv_incam_only=torch.tensor([False,True],device=args.device)),
        gt_cr_verts437=relative_vertices,gt_c_verts437=vertices,gt_j3d=joints)
    weights=dict(cr_j3d=500.,transl_c=1.,cr_verts=500.,j2d=1000.,verts2d=1000.,transl_w=1.,static_conf_bce=1.)
    if args.worker=='upstream':
        prediction.args.weights=OmegaConf.create(weights);prediction.weights=prediction.args.weights
        prediction.args.static_conf=OmegaConf.create(dict(vel_thr=.15));prediction.train();prediction.zero_grad(set_to_none=True)
    else:
        from motius.trainers.gvhmr import GVHMRTrainer
        trainer=GVHMRTrainer(bundle,body_model_path=args.body_model);bundle.train();bundle.zero_grad(set_to_none=True)
    torch.manual_seed(97)
    if args.worker=='upstream':
        result=prediction.forward(data,train=True)
        model=prediction.denoiser3d
        losses={k:v for k,v in result.items() if k.endswith("loss")}
    else:
        result=trainer.train_step(data);model=bundle.network
        losses={"loss":result["loss"],"simple_loss":result["gvhmr_simple"]}
        losses.update({k.removeprefix("gvhmr_"):v for k,v in result.items() if k.endswith("loss") and k!="loss"})
    result["loss"].backward()
    output["training"]={"losses":{k:v.detach().cpu() for k,v in losses.items()},
        "gradients":{k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None}}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for key in ['upstream','checkpoint','body-model','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    for key in ['upstream','checkpoint','body_model','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(exist_ok=True,parents=True)
    if args.worker:worker(args);return
    # Give the untouched source constructor the same licensed body file.
    destination=args.upstream/'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():destination.symlink_to(args.body_model)
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='GVHMR original core inference on nonzero precomputed inputs; video frontend not yet certified',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_sha256=digest(args.checkpoint),body_model_sha256=digest(args.body_model),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env=os.environ.copy();env['PYTHONPATH']=str(root)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--device',args.device]
        for key in ['upstream','checkpoint','body_model','output_dir']:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GVHMR core source comparisons exact')

if __name__=='__main__':main()
