"""Compare native KIMODO against a pinned original checkout with real local weights.

Workers are independent processes. Only adapter configuration paths are adjusted
to the supplied local Llama weights; source forward functions are unmodified.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from verify_intergen_source_parity import digest


def tensors(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:tensors(v) for k,v in value.items() if isinstance(v,(torch.Tensor,dict))}
    return value


def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    artifact=args.artifact
    os.environ.update(CHECKPOINT_DIR=str(artifact/'kimodo_checkpoint'),TEXT_ENCODERS_DIR=str(artifact/'text_encoders'),
        TEXT_ENCODER_MODE='local',TEXT_ENCODER='llm2vec',LOCAL_CACHE='true',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from kimodo.model.load_model import load_model
        from kimodo.constraints import Root2DConstraintSet
        model=load_model('Kimodo-SOMA-RP-v1',device=args.device)
        def generate(prompts,frames,**kwargs):
            return model(prompts,frames,return_numpy=False,**kwargs)
    else:
        from motius.pipelines.kimodo import KIMODOPipeline
        from motius.motion.conditioning.kimodo import Root2DConstraintSet
        pipeline=KIMODOPipeline.from_pretrained(str(artifact),device=args.device)
        model=pipeline.runtime
        def generate(prompts,frames,**kwargs):
            constraints=kwargs.pop('constraint_lst',None)
            return pipeline.generate(prompts,frames,constraints=constraints,return_numpy=False,**kwargs)
    output={}
    with torch.no_grad():
        features,lengths=model.text_encoder(['a person walks forward','a person raises both arms'])
        output['text_features']={'features':features.cpu(),'lengths':torch.as_tensor(lengths)}
    cases=[('single',13,'a person walks forward',30,False,False),
           ('batch',19,['a person walks forward','a person raises both arms'],[30,40],False,False),
           ('multiprompt',27,['a person walks forward','a person turns left'],[20,24],True,False),
           ('multiprompt_postprocess',27,['a person walks forward','a person turns left'],[20,24],True,True),
           ('root_control',31,'a person walks forward',30,False,True)]
    for name,seed,prompts,frames,multi,post in cases:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        constraints=[]
        if name=='root_control':
            constraints=[[Root2DConstraintSet(model.skeleton,
                torch.tensor([0,20],device=args.device),
                torch.tensor([[0.,0.],[.25,.5]],device=args.device))]]
        # Explicit sample count avoids the original repository's default-None multiprompt bug.
        result=generate(prompts,frames,num_denoising_steps=6,num_samples=1,
            multi_prompt=multi,post_processing=post,constraint_lst=constraints,
            progress_bar=lambda values:values)
        output[name]=tensors(result)
        print(args.worker,name,'complete',flush=True)
    torch.save(output,args.output_dir/f'{args.worker}.pt')
    metadata={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','numpy']}
    metadata.update(device=torch.cuda.get_device_name(),cuda=torch.version.cuda)
    (args.output_dir/f'{args.worker}_environment.json').write_text(json.dumps(metadata,indent=2)+'\n')


def compare(a,b,path,rows):
    import torch
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(path,a.keys(),b.keys())
        for k in a:compare(a[k],b[k],path+'/'+k,rows)
    else:
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),path
        rows.append(dict(tensor=path,shape=list(a.shape),bitwise_equal=torch.equal(a,b),
                         max_abs_error=(a.to(torch.float64)-b.to(torch.float64)).abs().max().item()))
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--worker',choices=['upstream','native'])
    args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='original KIMODO source and released SOMA-RP checkpoint, including the real LLM2Vec text encoder',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for p in (args.upstream/'kimodo').rglob('*.py')},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    # The same adapter weights must resolve to the artifact-local base instead of a gated Hub ID.
    base=args.artifact/'text_encoders/meta-llama/Meta-Llama-3-8B-Instruct'
    for directory in ['LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']:
        p=args.artifact/'text_encoders/McGill-NLP'/directory/'adapter_config.json'
        config=json.loads(p.read_text());config['base_model_name_or_path']=str(base);p.write_text(json.dumps(config,indent=2)+'\n')
    report['artifact_files']={str(p.relative_to(args.artifact)):digest(p)
        for p in args.artifact.rglob('*') if p.is_file() and '.cache' not in p.parts and p.suffix in ['.safetensors','.npy','.yaml','.json']}
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env=os.environ.copy();env['PYTHONPATH']=str(root)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),
            '--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--device',args.device,
            '--worker',kind],cwd=root,env=env,check=True)
    import torch
    a=torch.load(args.output_dir/'upstream.pt',weights_only=True)
    b=torch.load(args.output_dir/'native.pt',weights_only=True)
    try:compare(a,b,'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,native_files={str(p.relative_to(root)):digest(p)
        for folder in ['motius/models/kimodo','motius/pipelines/kimodo','motius/motion/representation/kimodo','motius/motion/body_models/kimodo']
        for p in (root/folder).rglob('*.py')})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All KIMODO source comparisons exact',flush=True)


if __name__=='__main__':main()
