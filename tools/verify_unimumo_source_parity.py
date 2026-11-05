"""Independent original UniMuMo inference/teacher-forcing tensor comparison."""
from __future__ import annotations
import argparse,json,os,random,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def source_generator_gradients(network):
    """Rename/split source parameter gradients without executing native math."""
    import re
    result={}
    names={'linear1':'music_ffn_in','linear2':'music_ffn_out','linear1_motion':'motion_ffn_in','linear2_motion':'motion_ffn_out',
        'norm1':'self_norm','norm1_motion':'motion_self_norm','norm2':'music_ffn_norm','norm2_motion':'motion_ffn_norm','norm_cross':'cross_norm',
        'self_attn':'self_attention','captioning_self_attn':'caption_attention','cross_attention':'cross_attention'}
    for key,parameter in network.named_parameters():
        if parameter.grad is None:continue
        value=parameter.grad.detach().cpu()
        if key.startswith('condition_provider.conditioners.description.output_proj.'):
            result['core.text_condition_projection.'+key.split('output_proj.',1)[1]]=value;continue
        if key.startswith('condition_provider.'):raise AssertionError('Frozen source text encoder unexpectedly received gradients: '+key)
        match=re.match(r'transformer.layers.(\d+).([^.]+).(.+)',key)
        if match:
            index,part,suffix=match.groups();prefix='core.generator.layers.'+index+'.'+names[part]+'.'
            if suffix in ['in_proj_weight','in_proj_bias']:
                field='weight' if suffix.endswith('weight') else 'bias'
                for projection,chunk in zip(['q','k','v'],value.chunk(3,dim=0)):result[prefix+projection+'_proj.'+field]=chunk
            else:result[prefix+suffix]=value
        else:
            for old,new in [('emb.','music_embeddings.'),('motion_emb.','motion_embeddings.'),('linears.','music_heads.'),('motion_linears.','motion_heads.'),('out_norm.','output_norm.')]:
                if key.startswith(old):result['core.generator.'+new+key[len(old):]]=value;break
            else:raise AssertionError('Unmapped source neural gradient: '+key)
    return result


def worker(args):
    import faulthandler
    faulthandler.enable(all_threads=True)
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from unimumo.models import UniMuMo
        package=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        # Redirect only a base-model filesystem location; all original
        # constructors, math, tokenizers and checkpoint keys are unchanged.
        package['music_motion_lm_config'].model.params.name=str(args.musicgen_base)
        model=UniMuMo(**package).to(args.device).eval()
        # Original from_checkpoint explicitly moves the Lightning child too,
        # updating its device metadata as well as its parameter storage.
        model.music_motion_lm=model.music_motion_lm.to(args.device)
        lm=model.music_motion_lm
        network=lm.model
    else:
        from motius.models.unimumo import UniMuMoBundle
        from motius.pipelines.unimumo import UniMuMoPipeline
        from motius.motion.conditioning.unimumo_pattern import DelayedPattern,restore_prediction_logits
        bundle=UniMuMoBundle.from_pretrained(str(args.artifact),local_files_only=True).to(args.device)
        model=UniMuMoPipeline(bundle);network=bundle.generator
    output={}
    # Nonzero normalized motion checks both source motion encoder and codec layout.
    motion=np.random.default_rng(113).normal(size=(2,60,263)).astype(np.float32)
    if args.worker=='upstream':
        normalized=torch.as_tensor(model.normalize_motion(motion),device=args.device)
        music_wave=torch.zeros(2,1,32000,device=args.device)
        with torch.no_grad():
            music_embedding=model.motion_vqvae.music_encoder(music_wave)
            raw=model.motion_vqvae.motion_encoder(normalized.transpose(1,2))
            _,motion_embedding=model.motion_vqvae.encode(music_wave,normalized)
            codes=model.encode_motion(motion)
    else:
        normalized=model.normalize_motion(torch.from_numpy(motion).to(args.device))
        with torch.no_grad():
            music_embedding=model._zero_audio_embeddings(2,60)
            raw=bundle.motion_codec.encoder(normalized.transpose(1,2))
            motion_embedding=bundle.motion_codec.encode_embeddings(normalized,music_embedding)
            codes=model.encode_motion(motion)
    output['encoding']={k:v.detach().cpu() for k,v in dict(normalized=normalized,music_embedding=music_embedding,raw_motion_embedding=raw,motion_embedding=motion_embedding,codes=codes).items()}
    for index,descriptions in enumerate([['piano solo <separation> a person walks'],['drums <separation> a person jumps','ambient <separation> slow dancing']]):
        batch=len(descriptions);seed(17)
        music_codes=torch.randint(0,2048,(batch,4,8),device=args.device);motion_codes=torch.randint(0,2048,(batch,4,8),device=args.device)
        hooks=[];intermediates={}
        if index==0:
            layer=network.transformer.layers[0] if args.worker=='upstream' else network.layers[0]
            hooks.append(layer.register_forward_pre_hook(lambda module,inputs:intermediates.update(layer_input=inputs[0].detach().cpu())))
            mapping={'norm1':'self_norm','self_attn':'self_attention','norm_cross':'cross_norm','cross_attention':'cross_attention',
                'norm2':'music_ffn_norm','norm2_motion':'motion_ffn_norm','linear1':'music_ffn_in','linear2':'music_ffn_out'}
            for original,native in mapping.items():
                module=getattr(layer,original if args.worker=='upstream' else native)
                def capture(module,inputs,result,key=original):
                    value=result[0] if isinstance(result,tuple) else result
                    intermediates[key]=value.detach().cpu()
                hooks.append(module.register_forward_hook(capture))
            hooks.append(layer.register_forward_hook(lambda module,inputs,result:intermediates.update(layer_output=result.detach().cpu())))
        if args.worker=='upstream':
            condition=lm.prepare_text_condition(descriptions,mode='music_motion')['description']
            predicted_music,predicted_motion=network.compute_predictions(music_codes,motion_codes,'music_motion',[],condition_tensors={'description':condition})
            values=dict(condition=condition[0],mask=condition[1],music_logits=predicted_music.logits,motion_logits=predicted_motion.logits,
                music_mask=predicted_music.mask,motion_mask=predicted_motion.mask)
        else:
            condition=model.text_condition(descriptions)
            pattern=DelayedPattern(8,(0,1,2,3))
            mi,_=pattern.build(music_codes,special_token=2048,valid_only=True);mo,_=pattern.build(motion_codes,special_token=2048,valid_only=True)
            music_logits,motion_logits=network(mi,mo,condition=condition[0],condition_mask=condition[1])
            music_logits,music_mask=restore_prediction_logits(music_logits,8,pattern.delays);motion_logits,motion_mask=restore_prediction_logits(motion_logits,8,pattern.delays)
            values=dict(condition=condition[0],mask=condition[1],music_logits=music_logits,motion_logits=motion_logits,music_mask=music_mask,motion_mask=motion_mask)
        # Invalid future logits are NaN by source contract; record masks and
        # compare every valid value without replacing any executed operation.
        for name in ['music','motion']:
            key=name+'_logits';values[key]=values[key][values[name+'_mask']]
        for handle in hooks:handle.remove()
        output[f'teacher_forcing_{index}']={k:v.detach().cpu() for k,v in values.items()}
        if index==0:output['intermediates']=intermediates
    descriptions=['electronic music <separation> a person dances']
    for index,temperature in enumerate([0.,1.]):
        seed(41+index)
        if args.worker=='upstream':
            music,motion=lm.generate_sample(batch={'text':descriptions,'music_code':None,'motion_code':None},duration=.24,
                conditional_guidance_scale=4.,temperature=temperature,return_result_only=True)
        else:
            music,motion=model.generate_codes(descriptions,timesteps=12,mode='music_motion',guidance_scale=4.,temperature=temperature,top_k=250,
                generator=torch.Generator(device=args.device).manual_seed(41+index))
        with torch.no_grad():
            if args.worker=='upstream':
                waveform=model.music_vqvae.decode(music)
                normalized=model.motion_vqvae.decode_from_code(music,motion)
                physical=torch.as_tensor(model.denormalize_motion(normalized.detach().cpu().numpy()),device=args.device)
            else:
                waveform=model.decode_audio(music);physical=model.decode_motion(music,motion)
        output[f'generation_{index}']={k:v.detach().cpu() for k,v in dict(music_codes=music,motion_codes=motion,waveform=waveform,motion=physical).items()}
    # Caption context and decoded captions exercise the separate self-attention
    # branches and pretrained T5 captioner, including original grad flags.
    seed(57);music_codes=torch.randint(0,2048,(1,4,10),device=args.device);motion_codes=torch.randint(0,2048,(1,4,10),device=args.device)
    for modality in ['music','motion']:
        if args.worker=='upstream':
            condition=lm.prepare_text_condition(['<separation>'],mode='music_motion')
            with torch.no_grad():
                context=network.get_music_motion_context(music_codes,motion_codes,[],modality+'_caption',condition_tensors=condition)
                caption=lm.text_model.generate_caption(context,modality+'_caption')
        else:
            condition=model.text_condition(['<separation>'])
            pattern=DelayedPattern(10,(0,1,2,3));mi,_=pattern.build(music_codes,special_token=2048,valid_only=True);mo,_=pattern.build(motion_codes,special_token=2048,valid_only=True)
            if modality=='music':mo[:]=2048
            else:mi[:]=2048
            with torch.no_grad():context=network.forward_features(mi,mo,condition=condition[0],condition_mask=condition[1],caption_mode=True)
            caption=model.caption(music_codes if modality=='music' else motion_codes,modality=modality)
        output['caption_'+modality]={'context':context.detach().cpu(),'text':{str(i):torch.tensor(list(text.encode('utf8')),dtype=torch.uint8) for i,text in enumerate(caption)}}
    # The native trainer accepts encoded conditions. Compare its published
    # objective/gradient graphs with independent original modules, then keep
    # raw text-dropout/data alignment protocol explicitly outside this certificate.
    for stage in ['train_music_motion','train_caption']:
        seed(73);music_codes=torch.randint(0,2048,(2,4,8),device=args.device);motion_codes=torch.randint(0,2048,(2,4,8),device=args.device)
        raw=torch.randn(2,6,768,device=args.device);mask=torch.tensor([[[1,1,1,0,0,0],[0,0,0,1,1,1]]]*2,device=args.device,dtype=torch.bool)
        if args.worker=='upstream':
            model.zero_grad(set_to_none=True);network.requires_grad_(stage=='train_music_motion');network.train()
            conditioner=network.condition_provider.conditioners['description'];conditioner.t5.requires_grad_(False).eval()
            lm.text_model.requires_grad_(stage=='train_caption').train(stage=='train_caption')
            if stage=='train_caption':
                for layer in network.transformer.layers:layer.captioning_self_attn.requires_grad_(True)
        else:
            from motius.trainers.unimumo import UniMuMoTrainer
            bundle.zero_grad(set_to_none=True);trainer=UniMuMoTrainer(bundle,stage=stage);bundle.train()
        seed(79)
        if args.worker=='upstream':
            condition=(conditioner.output_proj(raw),mask)
            if stage=='train_music_motion':
                pm,po=network.compute_predictions(music_codes,motion_codes,'music_motion',[],condition_tensors={'description':condition})
                music_loss,_=lm.compute_cross_entropy(pm.logits,music_codes,pm.mask)
                motion_loss,_=lm.compute_cross_entropy(po.logits,motion_codes,po.mask)
                loss=music_loss*.85+motion_loss*.15
            else:
                context=network.get_music_motion_context(music_codes,motion_codes,[],'motion_caption',condition_tensors={'description':condition})
                loss=lm.text_model(['a person walks','a person jumps'],context,'motion_caption')
        else:
            batch=dict(music_codes=music_codes,motion_codes=motion_codes,condition=raw,condition_mask=mask)
            if stage=='train_caption':
                encoded=model.tokenizer(['a person walks','a person jumps'],padding=True,return_tensors='pt',return_attention_mask=True)
                batch.update(caption_mode='motion',labels=encoded.input_ids,decoder_attention_mask=encoded.attention_mask)
            loss=trainer.train_step(batch)['loss']
        loss.backward()
        if args.worker=='upstream':
            gradients=source_generator_gradients(network)
            for key,parameter in lm.text_model.named_parameters():
                if parameter.grad is None:continue
                destination='captioner.'+key[len('model.'):] if key.startswith('model.') else 'core.caption_context_projection.'+key[len('context_proj.'):]
                gradients[destination]=parameter.grad.detach().cpu()
        else:gradients={key:parameter.grad.detach().cpu() for key,parameter in bundle.named_parameters() if parameter.grad is not None}
        output[stage]={'loss':loss.detach().cpu(),'gradients':gradients}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','checkpoint','musicgen-base','hf-home','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    keys=['upstream','artifact','checkpoint','musicgen_base','hf_home','output_dir']
    for key in keys:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='Original UniMuMo full checkpoint and native released artifact: motion codec, conditioned logits, greedy/stochastic generation and audio/motion decode',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        original_checkpoint_sha256=digest(args.checkpoint),musicgen_base_sha256=digest(args.musicgen_base/'state_dict.bin'),
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.rglob('*') if p.is_file() and '.cache' not in p.parts and 'benchmarks' not in p.parts and p.suffix in ['.safetensors','.npy','.json','.model']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--device',args.device]
        for key in keys:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=args.upstream if kind=='upstream' else root,env={**os.environ,'PYTHONPATH':str(root),'HF_HOME':str(args.hf_home),
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONFAULTHANDLER':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1' if kind=='upstream' else '0'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    folders=['motius/models/unimumo','motius/pipelines/unimumo','motius/trainers/unimumo','motius/motion/conditioning/unimumo_pattern.py']
    report.update(complete=True,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p) for folder in folders for p in (list((root/folder).rglob('*.py')) if (root/folder).is_dir() else [root/folder])})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All UniMuMo source comparisons exact')

if __name__=='__main__':main()
