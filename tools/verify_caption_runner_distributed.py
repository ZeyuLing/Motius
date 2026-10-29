"""Released native caption/token networks through the shared two-GPU Runner."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['tm2t','t2mgpt'],required=True)
    parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.artifact=args.artifact.resolve();args.output_dir=args.output_dir.resolve()
    import torch
    from torch.utils.data import Dataset
    from mmengine import Config
    from motius.registry import DATASETS
    from motius.runner import AccelerateRunner
    from verify_motionstreamer_source import hashes
    import motius.hooks
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    if args.method=='t2mgpt':
        import motius.models.t2mgpt,motius.trainers.t2mgpt
        meta=json.loads((args.artifact/'t2mgpt_config.json').read_text())
        model=dict(type='T2MGPTBundle',config=meta['config'],vq_weights_path=str(args.artifact/'vq.safetensors'),gpt_weights_path=str(args.artifact/'gpt.safetensors'),clip_weights_path=str(args.artifact/'clip.safetensors'),mean_path=str(args.artifact/'Mean.npy'),std_path=str(args.artifact/'Std.npy'),device='cpu')
        trainer=dict(type='T2MGPTTrainer',if_maxtest=False,pkeep=.5);root='gpt';vocabulary_size=0
    else:
        import motius.models.tm2t,motius.trainers.tm2t
        from motius.configuration.tm2t import vocabulary_metadata
        vocabulary_size,_,_=vocabulary_metadata(args.artifact)
        model=dict(type='TM2TBundle',artifact_dir=str(args.artifact),device='cpu')
        trainer=dict(type='TM2TTrainer',label_smoothing=True);root='transformer'

    @DATASETS.register_module()
    class NativeCaptionAuditDataset(Dataset):
        def __len__(self):return 32
        def __getitem__(self,index):
            generator=torch.Generator().manual_seed(71+index)
            if args.method=='t2mgpt':
                tokens=torch.randint(0,512,(11,),generator=generator);tokens[10]=512
                return dict(tokens=tokens,token_lens=torch.tensor(10),text=['A person walks.','A person raises both arms.'][index%2])
            motion=torch.randint(0,1024,(9,),generator=generator);motion[0]=1024;motion[-1]=1025
            text=torch.randint(0,vocabulary_size,(8,),generator=generator)
            return dict(motion_tokens=motion,text_tokens=text)

    cfg=Config(dict(work_dir=str(args.output_dir/'native'),model=model,trainer=trainer,
        optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-6),
        train_dataloader=dict(batch_size=2,num_workers=0,shuffle=True,drop_last=True,dataset=dict(type='NativeCaptionAuditDataset')),
        train_cfg=dict(by_epoch=False,max_iters=3,max_grad_norm=1.),
        accelerator=dict(mixed_precision='no',gradient_accumulation_steps=1,step_scheduler_with_optimizer=False,ddp_kwargs=dict(find_unused_parameters=args.method=='t2mgpt')),
        cuda_precision=dict(allow_tf32=False,float32_matmul_precision='highest'),
        sdpa_backends=dict(flash=False,mem_efficient=False,math=True),
        default_hooks=dict(checkpoint=dict(type='CheckpointHook',interval=3,max_keep_ckpts=1,save_last=True)),
        seed=37))
    runner=AccelerateRunner.from_cfg(cfg)
    frozen={name:hashes(getattr(runner.bundle,name)) for name in runner.bundle._frozen_modules}

    class Check:
        def after_run(self):
            assert runner.global_step==3
            neural=runner.accelerator.unwrap_model(getattr(runner.bundle,root))
            state=hashes(neural)
            replicas=[None]*runner.accelerator.num_processes
            torch.distributed.all_gather_object(replicas,state)
            assert all(value==replicas[0] for value in replicas),'Trained neural replicas diverged'
            for name,before in frozen.items():assert hashes(getattr(runner.bundle,name))==before,'Frozen network changed: '+name
            if args.method=='t2mgpt':
                fixed=[p for name,p in neural.named_parameters() if 'pos_embed.embed' in name]
                assert fixed and all(not p.requires_grad and p.grad is None for p in fixed)
            if runner.accelerator.is_main_process:
                path=args.output_dir/'native/checkpoint-iter_3/model.pt';assert path.is_file()
                checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                assert all(name in checkpoint for name in runner.bundle._save_ckpt_modules)
                report=dict(complete=True,scope=args.method+' released neural networks, native Trainer and actual shared AccelerateRunner, three two-GPU updates with distinct rank-local prepared data. All trained neural hashes synchronized, frozen networks unchanged, complete trainable/frozen checkpoint roots present. Synthetic prepared-token fixture; not independent original distributed training or full dataset protocol.',world_size=runner.accelerator.num_processes,updates=3,neural_tensors=len(state),frozen_roots=list(frozen),all_replica_neural_hashes_equal=True)
                args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/'source_parity.json').write_text(json.dumps(report,indent=2)+'\n')
            runner.accelerator.wait_for_everyone()
    hook=Check();hook.runner=runner;runner.hooks.append(hook)
    runner.train()

if __name__=='__main__':main()
