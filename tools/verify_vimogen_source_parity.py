"""Original ViMoGen neural/text/inference graphs versus native composition."""
from __future__ import annotations
import argparse,ast,json,os,random,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def source_function(args,name,namespace):
    """Compile an unchanged nested source function without importing its app."""
    path=args.upstream/'train_eval_vimogen.py';tree=ast.parse(path.read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


def worker(args):
    import numpy as np
    import torch
    from tqdm import tqdm
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dtype=torch.bfloat16;device=torch.device(args.device)
    def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from models.transformer import get_transformer3d
        from models.transformer.wan.modules.t5 import T5EncoderModel
        from models.transformer.utils import randn_tensor
        import importlib.util
        spec=importlib.util.spec_from_file_location('original_vimogen_scheduler',args.upstream/'trainer/scheduler.py')
        schedule_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(schedule_module)
        FlowMatchScheduler=schedule_module.FlowMatchScheduler
        from utils import smooth_motion_rep,maybe_corrupt_ref_motion
        network=get_transformer3d('wanvideotm2m_1.3b',True,in_channel=276,base_repo=str(args.artifact/'wan'),strict=True,
            model_kwargs=dict(dense_interval=1,rope_mode='naive',force_no_sincos_embed=True,i2v_mode=0,in_channels=276,ref_motion_dim=138,load_path=str(args.artifact/'model.pt'))).to(device=device,dtype=dtype).eval()
        text=T5EncoderModel(text_len=512,dtype=dtype,device=device,checkpoint_path=str(args.artifact/'wan/models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(args.artifact/'wan/google/umt5-xxl'))
        class Logger:
            def info(self,*a,**kw):pass
            def warning(self,*a,**kw):pass
        namespace=dict(torch=torch,tqdm=tqdm,FlowMatchScheduler=FlowMatchScheduler,randn_tensor=randn_tensor,
            smooth_motion_rep=smooth_motion_rep,to_eval_mode=network.eval,to_train_mode=network.train)
        generate=source_function(args,'generate_pipe',namespace)
    else:
        from motius.models.vimogen import ViMoGenBundle
        from motius.pipelines.vimogen import ViMoGenPipeline
        from motius.diffusion.vimogen import FlowMatchScheduler
        bundle=ViMoGenBundle(artifact_dir=str(args.artifact),device=args.device,dtype='bf16')
        pipeline=ViMoGenPipeline(bundle);network=bundle.model
    output={};captions=['A person walks forward.','A person raises both arms.']
    with torch.no_grad():
        if args.worker=='upstream':
            rows=text(captions,device)
            null=torch.load(args.artifact/'context_null_padded.pth',map_location=device,weights_only=True)
            if null.ndim==2:null=null[None]
            count=max(null.shape[1],*[len(row) for row in rows])
            context=torch.zeros(2,count,rows[0].shape[-1],device=device,dtype=dtype)
            for i,row in enumerate(rows):context[i,:len(row)]=row
        else:context=pipeline.encode_texts(captions)
    output['text']=context.cpu()
    # Text is tested with the real UMT5. Release GPU memory before neural
    # gradients; all subsequent cases use the independently computed features.
    if args.worker=='upstream':del text
    else:bundle.text_encoder=None
    import gc
    gc.collect();torch.cuda.empty_cache()
    mean=torch.from_numpy(np.load(args.artifact/'assets/meta/mean.npy')).to(device)
    std=torch.from_numpy(np.load(args.artifact/'assets/meta/std.npy')).to(device)
    for index,lengths in enumerate([[40],[60,40]]):
        frames=max(lengths);batch=len(lengths);prompt=context[:batch]
        if args.worker=='upstream':
            latent=torch.zeros(batch,frames,276,device=device,dtype=dtype)
            mask=torch.zeros(batch,frames,device=device,dtype=dtype)
            for row,length in enumerate(lengths):mask[row,:length]=1
            reference=torch.zeros(batch,frames,138,device=device,dtype=dtype)
            normalized=generate(network,prompt,null.repeat(batch,1,1).to(dtype),latent,mask,reference,mask.clone(),
                num_inference_steps=50,cfg_scale=5.,device=device,dtype=dtype,scheduler=FlowMatchScheduler(),seed=31+index,
                logger=Logger(),condition_on_text=True,attend_to_text_mask=torch.ones(batch,device=device,dtype=torch.bool))
            values=[(normalized[i,:length]*std+mean).float().cpu() for i,length in enumerate(lengths)]
        else:values=pipeline.generate_motion276_from_embeddings(prompt,lengths,seed=31+index)
        output[f'generation_{index}']={str(i):value for i,value in enumerate(values)}
    # Exact source objective statements are compiled directly, including noise
    # draws, scheduler weights, masks and global-channel upweighting.
    seed(73);motion=torch.randn(2,32,276,device=device,dtype=dtype)
    mask=torch.ones(2,32,device=device,dtype=dtype);mask[1,24:]=0
    reference=torch.zeros(2,32,138,device=device,dtype=dtype);reference[1,:24]=torch.randn(24,138,device=device,dtype=dtype)
    dims=torch.ones(2,276,device=device);dims[1,:6]=0
    batch=dict(motion=motion,motion_mask=mask,prompt_emb=context,ref_motion=reference,ref_motion_mask=mask.clone(),
        motion_dim_mask=dims,attend_to_text_mask=torch.tensor([True,False],device=device))
    network.train().requires_grad_(True);network.zero_grad(set_to_none=True);seed(79)
    if args.worker=='upstream':
        tree=ast.parse((args.upstream/'train_eval_vimogen.py').read_text())
        # These top-level training statements reside inside the training loop.
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='global_step')
        branch=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='is_training')
        begin=next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='wan_scheduler.set_timesteps')
        end=next(i for i,n in enumerate(branch.body) if i>begin and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='no_sync' for t in n.targets))
        statements=branch.body[begin:end]
        assert any(isinstance(n,ast.With) for n in statements)
        scope=dict(torch=torch,wan_scheduler=FlowMatchScheduler(),latents=motion,latents_mask=mask,prompt_emb=context,
            ref_latents=reference,ref_latents_mask=batch['ref_motion_mask'],motion_dim_mask=dims,
            attend_to_text_mask=batch['attend_to_text_mask'],device=device,dtype=dtype,model=network,
            timer=SimpleNamespace(forward=nullcontext()),avg_loss_dict={'loss':0.,'loss_text':0.,'loss_ref_motion':0.})
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(args.upstream/'train_eval_vimogen.py'),'exec'),scope)
        loss=scope['loss']
    else:
        from motius.trainers.vimogen import ViMoGenTrainer
        trainer=ViMoGenTrainer(bundle);loss=trainer.train_step(batch)['loss']
    loss.backward()
    output['training']={'loss':loss.detach().cpu(),'gradients':{k:p.grad.detach().cpu() for k,p in network.named_parameters() if p.grad is not None}}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='Original ViMoGen networks, real UMT5, unchanged source generate_pipe and training objective versus native bundle/pipeline/trainer',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        reference_extraction='Unchanged nested generate_pipe and original objective AST statements; only logging/mode closures are supplied independently of FSDP application setup',
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.rglob('*') if p.is_file() and '.cache' not in p.parts and p.suffix in ['.pt','.pth','.npy','.json','.model']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--device',args.device]
        for key in ['upstream','artifact','output_dir']:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    folders=['motius/models/vimogen','motius/pipelines/vimogen','motius/trainers/vimogen','motius/diffusion/vimogen.py','motius/motion/conditioning/vimogen.py']
    report.update(complete=True,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p) for folder in folders for p in (list((root/folder).rglob('*.py')) if (root/folder).is_dir() else [root/folder])})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All ViMoGen source comparisons exact')

if __name__=='__main__':main()
