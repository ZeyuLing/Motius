"""Compare VerMo released-weight inference with the pre-migration package snapshot.

This is migration evidence, not an independent original-repository certificate.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    if args.worker=='reference':sys.path.insert(0,str(args.reference))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from motius.models.vermo import VermoBundle
    from motius.pipelines.vermo import VermoPipeline
    bundle=VermoBundle.from_pretrained(str(args.artifact),device=args.device)
    pipeline=VermoPipeline(bundle)
    processor=bundle.processor if args.worker=='reference' else pipeline.processor
    output={}
    torch.manual_seed(71)
    for length in [16,25]:
        motion=torch.randn(length,138,device=args.device)*.01
        motion[:,:3]=torch.arange(length,device=args.device)[:,None]*.001
        if args.worker=='reference':
            from motius.models.vermo.task_utils.task_lib.text_motion_tasks.motion2caption import Motion2Caption
        else:
            from motius.motion.conditioning.vermo.task_lib.text_motion_tasks.motion2caption import Motion2Caption
        batch=dict(task=[Motion2Caption()],num_person=[1],motion=[motion],caption=[''])
        # Inspect real codec and prompt formatting in addition to final captions.
        with torch.no_grad():
            prepared=processor.process_train(batch)
            language={k:v for k,v in prepared.items() if k in ['input_ids','attention_mask']}
            logits=bundle.lm(**language,use_cache=False).logits
            captions=pipeline.infer_m2t([motion],[length],input_representation='vermo138',max_new_tokens=24)
        output[str(length)]={k:v.cpu() for k,v in prepared.items() if isinstance(v,torch.Tensor)}
        output[str(length)]['logits']=logits.cpu()
        output[str(length)]['caption']=torch.tensor(list(captions[0].encode('utf-8')),dtype=torch.uint8)
    bundle.lm.requires_grad_(True).train();bundle.lm.zero_grad(set_to_none=True)
    torch.manual_seed(411)
    tokens=torch.randint(0,bundle.lm.config.vocab_size,(2,11),device=args.device)
    labels=tokens.clone();labels[:,:3]=-100;labels[1,-2:]=-100
    weights=torch.ones_like(labels,dtype=torch.float);weights[:,3]=2.
    boundary=torch.zeros_like(tokens,dtype=torch.bool);boundary[:,3]=True
    torch.manual_seed(53)
    if args.worker=='reference':
        os.environ['VERMO_LM_LOSS_CHUNK_SIZE']='4'
        result=bundle.lm(input_ids=tokens,labels=labels,loss_token_weights=weights,loss_boundary_mask=boundary,use_cache=False)
        loss=result.loss
    else:
        from motius.trainers.vermo import VermoTrainer
        trainer=VermoTrainer(bundle,processor=processor,loss_chunk_size=4)
        result=trainer.train_step(dict(input_ids=tokens,labels=labels,loss_token_weights=weights,loss_boundary_mask=boundary))
        loss=result['loss']
    loss.backward()
    output['training']=dict(loss=loss.detach().cpu(),gradients={k:p.grad.detach().cpu() for k,p in bundle.lm.named_parameters() if p.grad is not None})
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['reference','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['reference','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for key in ['reference','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'migration_parity.json'
    report=dict(complete=False,comparison='Pre-migration package versus native VerMo with full released motion codec and Llama weights',evidence_kind='migration_snapshot_not_independent_original_source',reference_files={str(p.relative_to(args.reference)):digest(p) for p in (args.reference/'motius/models/vermo').rglob('*.py')},artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.rglob('*') if p.is_file() and '.cache' not in p.parts},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['reference','native']:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'reference.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All VerMo migration comparisons exact')

if __name__=='__main__':main()
