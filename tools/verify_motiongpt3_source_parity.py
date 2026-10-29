"""Independent MotionGPT3 released-source inference comparison."""
from __future__ import annotations
import argparse,json,os,random,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from omegaconf import OmegaConf
        from motGPT.archs.motgpt_lm import MLM
        from motGPT.archs.mld_vae import MldVae
        config=OmegaConf.merge(OmegaConf.load(args.artifact/'configs/default.yaml'),OmegaConf.load(args.artifact/'configs/test.yaml'))
        for folder in ['lm','vae']:
            for path in sorted((args.artifact/'configs'/folder).glob('*.yaml')):
                OmegaConf.update(config,f'{folder}.{path.stem}',OmegaConf.load(path),force_add=True)
        config=OmegaConf.merge(config,OmegaConf.load(args.artifact/'configs/assets.yaml'))
        vc=OmegaConf.to_container(config.model.params.motion_vae.params,resolve=True);vc['datatype']='humanml3d'
        lc=OmegaConf.to_container(config.model.params.lm.params,resolve=True)
        lc['model_path']=str(args.artifact/'deps/mot-gpt2');lc['diffhead']['target']='motGPT.diffusion.diffloss.DiffLoss'
        vq=MldVae(**vc);lm=MLM(**lc,vae_latent_channels=vq.latent_dim,vae_latent_size=vq.latent_size)
        state=torch.load(args.artifact/'motiongpt3.ckpt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
        for name,model in [('lm',lm),('vae',vq)]:
            model.load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')},strict=True)
            model.to(args.device).eval()
        decoder=lm
    else:
        from motius.models.motiongpt3 import MotionGPT3Bundle
        from motius.pipelines.motiongpt3 import MotionGPT3Pipeline
        bundle=MotionGPT3Bundle(artifact_dir=str(args.artifact),device=args.device)
        vq,lm=bundle.vae,bundle.lm;pipeline=MotionGPT3Pipeline(bundle);decoder=pipeline.decoder
    output={}
    mean=torch.from_numpy(np.load(args.artifact/'assets/meta/mean.npy')).to(args.device)
    std=torch.from_numpy(np.load(args.artifact/'assets/meta/std.npy')).to(args.device)
    for index,lengths in enumerate([[40],[60,40]]):
        text=['A person walks forward.','A person raises both arms.'][:len(lengths)]
        seed(31+index)
        with torch.no_grad():
            conditioning=decoder.generate_conditional(text,lengths=lengths,stage='test',tasks=None)
            latents,mask=decoder.sample_tokens(conditioning,torch.device(args.device),temperature=1.,cfg=7.5,vae_mean_std_inv=vq.mean_std_inv)
            motion=vq.decode(latents.reshape(len(lengths),vq.latent_size,-1).permute(1,0,2),lengths=lengths)
            mask=mask.to(device=motion.device,dtype=torch.bool)
            motion=torch.where(mask[:,None,None],torch.zeros_like(motion),motion)
        output[f'generation_{index}']={k:v.cpu() for k,v in dict(conditioning=conditioning,latents=latents,mask=mask,motion=motion,physical=motion*std+mean).items()}
    seed(73);motion=torch.randn(2,48,263,device=args.device);lengths=[48,40]
    with torch.no_grad():
        posterior=vq.encode_dist(motion,lengths);latents,distribution=vq.encode_dist2z(posterior)
        reconstruction=vq.decode(latents,lengths)
        captions=decoder.generate_conditional(motion_feats=motion,motion_encode_net=vq,lengths=lengths,task='m2t',stage='test',with_len=False)
    output['motion_encoding']={k:v.cpu() for k,v in dict(posterior=posterior,latents=latents,reconstruction=reconstruction).items()}
    output['captions']={str(i):torch.tensor(list(text.encode('utf8')),dtype=torch.uint8) for i,text in enumerate(captions)}
    # Published language objective includes source VAE reparameterization,
    # modality CE and latent diffusion. Compare actual released trainable grads.
    if args.worker=='native':
        from motius.trainers.motiongpt3 import MotionGPT3Trainer
        trainer=MotionGPT3Trainer(bundle)
    for index,dropout in enumerate([.1,1.]):
        seed(93);training_motion=torch.randn(2,40,263,device=args.device)
        tasks=[{'input':['Generate motion: <Caption_Placeholder>'],'output':['<Motion_Placeholder>'],'class':'t2m'}]*2
        batch=dict(motion=training_motion,text=['walk forward','jump once'],length=[40,32],tasks=tasks)
        lm.zero_grad(set_to_none=True)
        if args.worker=='upstream':
            lm.train();vq.eval();vq.requires_grad_(False);lm.guidance_uncondp=dropout
        else:
            bundle.diffusion_head.zero_grad(set_to_none=True);bundle.options['guidance_uncondp']=dropout;bundle.train()
        seed(79+index)
        if args.worker=='upstream':
            value=lm(batch['text'],training_motion,vq,batch['length'],tasks)
            cls,diffusion=value.loss,value.diff_loss;loss=cls+diffusion
        else:
            value=trainer.train_step(batch);cls,diffusion=value['language_loss'],value['diffusion_loss'];loss=value['loss']
        assert diffusion.item()>0,'Original diffusion path must execute, not silently skip'
        loss.backward()
        gradients={k:p.grad.detach().cpu() for k,p in lm.named_parameters() if p.grad is not None}
        if args.worker=='native':
            for key,parameter in bundle.diffusion_head.named_parameters():
                if parameter.grad is not None:gradients['fake_latent' if key=='fake_latent' else 'diffloss.'+key]=parameter.grad.detach().cpu()
        output[f'language_training_{index}']={'loss':loss.detach().cpu(),'classification':cls.detach().cpu(),'diffusion':diffusion.detach().cpu(),'gradients':gradients}
    # Stage-one VAE includes the unused re-encoding draw, preserving source RNG.
    vq.zero_grad(set_to_none=True);vq.requires_grad_(True);vq.train();seed(89)
    if args.worker=='upstream':
        from motGPT.losses.motgpt import MotLosses
        from omegaconf import OmegaConf
        z,distribution=vq.encode(training_motion,[40,32]);reconstruction=vq.decode(z,[40,32]);vq.encode(reconstruction,[40,32])
        reference=torch.distributions.Normal(torch.zeros_like(distribution.loc),torch.ones_like(distribution.scale))
        config=OmegaConf.create({'ABLATION':{'PREDICT_EPSILON':True},'LOSS':{'ABLATION':{'RECONS_LOSS':'l1_smooth'},
            'LAMBDA_FEATURE':1.,'LAMBDA_VELOCITY':.5,'LAMBDA_KL':1e-5}})
        loss=MotLosses(config,'vae',22).to(args.device).update(dict(m_ref=training_motion,m_rst=reconstruction,dist_m=distribution,dist_ref=reference))
    else:
        trainer=MotionGPT3Trainer(bundle,stage='vae');bundle.train();loss=trainer.train_step(dict(motion=training_motion,length=[40,32]))['loss']
    loss.backward()
    output['vae_training']={'loss':loss.detach().cpu(),'gradients':{k:p.grad.detach().cpu() for k,p in vq.named_parameters() if p.grad is not None}}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='Original MotionGPT3 source and released checkpoint: language conditioning, full latent diffusion, decoded motions, VAE and captions',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for folder in ['motGPT/archs','motGPT/diffusion','mot_code'] for p in (args.upstream/folder).rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.rglob('*') if p.is_file() and '.cache' not in p.parts},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--device',args.device]
        for key in ['upstream','artifact','output_dir']:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1' if kind=='upstream' else '0'},check=True)
    import torch,transformers
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    folders=['motius/models/motiongpt3','motius/trainers/motiongpt3','motius/pipelines/motiongpt3','motius/diffusion/motiongpt3','motius/motion/conditioning/motiongpt3.py','motius/configuration/motiongpt3.py','motius/configuration/motiongpt3_text_parameters.json','motius/motion/conditioning/motiongpt3_masks.py']
    report.update(complete=True,torch_version=torch.__version__,transformers_version=transformers.__version__,native_files={str(p.relative_to(root)):digest(p)
        for folder in folders for p in (list((root/folder).rglob('*.py')) if (root/folder).is_dir() else [root/folder])})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All MotionGPT3 source comparisons exact')

if __name__=='__main__':main()
