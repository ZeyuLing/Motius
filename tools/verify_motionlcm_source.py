"""Released MotionLCM weights against independent original neural and sampling code."""
from pathlib import Path
import argparse,ast,inspect,json,os,subprocess,sys
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import torch,numpy as np,tqdm
    from safetensors.torch import load_file
    from verify_prompthmr_video_parity import tensors
    from verify_motionstreamer_source import hashes
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    config=json.loads((args.artifact/'motionlcm_config.json').read_text())['config']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from mld.models.architectures.mld_clip import MldTextEncoder
        from mld.models.architectures.mld_vae import MldVae
        from mld.models.architectures.mld_denoiser import MldDenoiser
        from mld.utils.utils import get_guidance_scale_embedding
        from diffusers import LCMScheduler
        text=MldTextEncoder(str(args.artifact/'text_encoder/sentence-t5-large')).cuda().eval()
        vae=MldVae(**config['vae']).cuda().eval()
        denoiser=MldDenoiser(**config['denoiser']).cuda().eval()
        for module,name in [(vae,'vae'),(denoiser,'denoiser')]:module.load_state_dict(load_file(str(args.artifact/f'{name}.safetensors')),strict=True)
        tree=ast.parse((args.upstream/'mld/models/modeltype/mld.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_diffusion_reverse')
        namespace={'torch':torch,'inspect':inspect,'tqdm':tqdm,'get_guidance_scale_embedding':get_guidance_scale_embedding}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(args.upstream/'mld/models/modeltype/mld.py'),'exec'),namespace)
        reverse=namespace['_diffusion_reverse']
    else:
        from motius.pipelines.motionlcm import MotionLCMPipeline
        from motius.motion.conditioning.motionlcm_text import encode_text
        pipeline=MotionLCMPipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':'cuda'})
        text,vae,denoiser=pipeline.bundle.text_encoder,pipeline.bundle.vae,pipeline.bundle.denoiser
    output={'neural':{'vae':hashes(vae),'denoiser':hashes(denoiser)},'cases':{}}
    for i,(captions,lengths,steps) in enumerate([(['A person walks forward.'],[40],1),(['A person walks.','A person raises both arms.'],[48,40],2),([''],[40],4)]):
        with torch.no_grad():
            features=text(captions) if args.worker=='upstream' else encode_text(text,captions)
            torch.manual_seed(57+i)
            if args.worker=='upstream':
                scheduler=LCMScheduler(**config['scheduler'])
                context=NS(is_controlnet=False,do_classifier_free_guidance=False,guidance_scale=7.5,
                    scheduler=scheduler,denoiser=denoiser,latent_dim=config['vae']['latent_dim'],
                    cfg=NS(model=NS(scheduler=NS(num_inference_timesteps=steps))))
                latent=reverse(context,features)
                motion=vae.decode(latent,lengths)
                stats=[torch.from_numpy(np.load(args.artifact/f'{name}.npy')).cuda() for name in ['Mean','Std']]
                physical=motion*stats[1]+stats[0]
                values=[physical[index,:length] for index,length in enumerate(lengths)]
            else:
                values=pipeline.infer_t2m(captions,lengths,num_inference_steps=steps)
            output['cases'][str(i)]=tensors({'features':features,'physical':values,'rng':torch.cuda.get_rng_state()})
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native'])
    parser.add_argument('--reuse-results',action='store_true')
    args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    from verify_prompthmr_video_parity import compare_full
    import torch
    report={'complete':False,'scope':'Original released MotionLCM text-only graph: real SentenceT5, strict VAE/denoiser weights and original reverse method (executed unmodified AST), complete 1/2/4 step sampling, variable lengths, empty text and RNG. Excludes ControlNet and training.',
            'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),'tests':[]}
    path=args.output_dir/'source_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.reuse_results else ['upstream','native']):
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--worker',kind],cwd=ROOT,env=env,check=True)
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n');print('MotionLCM original full text-only inference exact')

if __name__=='__main__':main()
