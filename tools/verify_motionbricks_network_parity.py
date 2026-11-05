"""Compare released MotionBricks neural components against independent original source."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=args.artifact/'motionbricks_checkpoint'
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream/'motionbricks'))
        from motionbricks.helper.pl_util import load_motion_rep
        from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
        from motionbricks.motion_backbone.neural_modules.root_backbone import root_backbone_network
        from motionbricks.vqvae.neural_modules.vqvae import VQVAE
        from motionbricks.helper.data_training_util import extract_feature_from_motion_rep,convert_sparse_cond_to_dense_cond_if_needed
        cfg=OmegaConf.load(root/'motionbricks_pose/version_1/hparams.yaml')
        cfg.skeleton.folder=str(root/'motionbricks_pose/version_1/skeleton');cfg.motion_rep.stats.folder=str(root/'motionbricks_pose/version_1/stats/motion')
        representation=load_motion_rep(cfg)
        pose=pose_backbone_network(representation,cfg.model.backbone_network.args)
        root_cfg=OmegaConf.load(root/'motionbricks_root/version_1/hparams.yaml')
        global_root=root_backbone_network(root_cfg.model.backbone_network.args,representation)
        options=OmegaConf.to_container(cfg.model.pose_vqvae_network,resolve=True);options.pop('_target_');options.pop('motion_rep')
        tokenizer=VQVAE(motion_rep=representation.dual_rep.local_motion_rep,**options)
        for name,module,prefix in [('vqvae',tokenizer,'pose_net.'),('pose',pose,'backbone_net.'),('root',global_root,'backbone_net.')]:
            state=torch.load(root/('motionbricks_'+name)/'version_1/checkpoints/model-step=2000000.ckpt',map_location='cpu',weights_only=False,mmap=True)['state_dict']
            module.load_state_dict({key[len(prefix):]:value for key,value in state.items()},strict=True)
            module.cuda().eval()
    else:
        from motius.models.motionbricks import MotionBricksBundle
        from motius.configuration.motionbricks import build_representation
        from motius.motion.conditioning.motionbricks import extract_feature_from_motion_rep,convert_sparse_cond_to_dense_cond_if_needed
        bundle=MotionBricksBundle(args.artifact,device='cuda')
        pose=bundle.pose;global_root=bundle.root;tokenizer=None
        representation=build_representation(args.artifact)
    results={};local=representation.dual_rep.local_motion_rep
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items() if isinstance(v,(torch.Tensor,dict))}
        return value
    for case,(batch_size,tokens,training) in enumerate([(1,8,False),(2,12,False),(2,10,True)]):
        torch.manual_seed(117+case);frames=tokens*4
        pose.train(training);global_root.train(training)
        if tokenizer is not None:tokenizer.eval()
        else:
            for name in ['encoder','decoder','quantizer']:getattr(bundle,name).eval()
        lengths=torch.tensor([tokens,tokens-2][:batch_size],device='cuda')[:,None]
        features=torch.randn(batch_size,frames,len(local.indices['all']),device='cuda')
        cond=features.clone()
        has_cond=torch.zeros(batch_size,frames,device='cuda',dtype=torch.bool);has_cond[:,:4]=True;has_cond[:,-4:]=True
        dense,mask=convert_sparse_cond_to_dense_cond_if_needed(cond,has_cond,frames)
        encoder_feature=extract_feature_from_motion_rep(features,local,'joint_positions_and_rotations_and_hip_height')
        target_feature=extract_feature_from_motion_rep(dense,local,'joint_positions_and_rotations_and_hip_height')
        external=torch.randn(batch_size,frames,2,device='cuda')
        with torch.no_grad():
            if args.worker=='upstream':
                codes=tokenizer.encode_into_idx(features,fetch_overall_indices=False)
                reconstruction=tokenizer(features,cond,has_cond,external)
                decoding=tokenizer.forward_decoder(codes,cond,has_cond,external,use_overall_indices=False)
            else:
                codes=bundle.encode_tokens(encoder_feature)
                reconstruction=bundle.reconstruct(encoder_feature,target_cond=target_feature,has_target_cond=mask,external_cond=external)
                reconstruction['l_commit']=torch.nn.functional.mse_loss(reconstruction['encoder_features'],reconstruction['code_vectors'])
                decoding=bundle.decode_tokens(codes,target_cond=target_feature,has_target_cond=mask,external_cond=external)
        outputs=dict(codes=codes,reconstruction=reconstruction['recon_state'],commit=reconstruction['l_commit'],decoding=decoding['recon_state'])
        pose_inputs=dict(pose_tokens=codes,local_root_values=torch.randn(batch_size,frames,4,device='cuda'),
            pose_cond=target_feature,has_pose_cond=mask,num_tokens=lengths,
            text_embeddings=torch.randn(batch_size,4096,device='cuda'),has_text_embeddings=torch.ones(batch_size,device='cuda',dtype=torch.bool))
        torch.manual_seed(55+case)
        prediction=pose(**pose_inputs)
        outputs.update(pose_logits=prediction['pose_logits'])
        root_inputs=dict(global_root_values=torch.randn(batch_size,8,5,device='cuda'),has_global_root_values=torch.ones(batch_size,8,device='cuda',dtype=torch.bool),
            local_root_values=torch.randn(batch_size,8,4,device='cuda'),has_local_root_values=torch.ones(batch_size,8,device='cuda',dtype=torch.bool),
            poses=target_feature[:,[0,1,2,3,frames-4,frames-3,frames-2,frames-1]],has_poses=torch.ones(batch_size,8,device='cuda',dtype=torch.bool),
            num_tokens=torch.full((batch_size,1),global_root.MASKED_NUM_TOKENS,device='cuda',dtype=torch.long),
            text_embeddings=pose_inputs['text_embeddings'],has_text_embeddings=pose_inputs['has_text_embeddings'])
        torch.manual_seed(71+case)
        if args.worker=='upstream':prediction=global_root(**root_inputs,groundtruth_num_tokens=lengths if training else None)
        else:
            if training:prediction=bundle.predict_root(**root_inputs,chosen_token=lengths.reshape(-1)-global_root.args['min_tokens'])
            else:
                from motius.pipelines.motionbricks.sampling import select_root_token
                context=bundle.predict_root(**root_inputs,phase='length')
                chosen,logits=select_root_token(global_root,context['num_token_logits'],root_inputs['num_tokens'])
                context['num_token_logits']=logits
                prediction=bundle.predict_root(phase='motion',context=context,chosen_token=chosen)
        outputs.update(prediction)
        if training:
            loss=outputs['pose_logits'].square().mean()+prediction['pred_global_root_values'].square().mean()+prediction['num_token_logits'].square().mean()
            loss.backward();outputs['graph_loss']=loss
            outputs['gradients']={prefix+'.'+key:value.grad for prefix,network in [('pose',pose),('root',global_root)] for key,value in network.named_parameters() if value.grad is not None}
        results[str(case)]=cpu(outputs)
    torch.save(results,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'network_parity.json';report=dict(complete=False,comparison='Independent original MotionBricks released pose/root/VQ networks versus native neural roots and separate root-length policy',scope='Neural graphs and synthetic graph-objective gradients only; full navigation and original training objectives require separate certificates',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream/'motionbricks' if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All MotionBricks neural comparisons exact')

if __name__=='__main__':main()
