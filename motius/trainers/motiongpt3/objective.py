"""MotionGPT3 task preparation and released modality loss equations."""
import random
import torch
import torch.nn.functional as F


def prepare_language_batch(processor,vae,texts,motion,lengths,tasks):
    processor.tokenizer.padding_side='right'
    with torch.no_grad():
        # Preserve the published random draw, even though all three choices match.
        random.choice(['supervised','supervised','supervised'])
        inputs,outputs,modes=processor.template_fulfill(tasks,lengths,texts)
        labels=[source+' \n '+target+processor.tokenizer.eos_token for source,target in zip(inputs,outputs)]
    encoded=processor.tokenizer(labels,padding='max_length',max_length=processor.max_length,
        truncation=True,return_attention_mask=True,return_tensors='pt')
    ids=encoded.input_ids.to(motion.device);attention=encoded.attention_mask.to(motion.device)
    input_is_motion=ids==processor.motion_holder_id
    output_is_motion=ids==processor.motion_holder_id_out
    types=torch.zeros_like(ids);is_motion=ids>=processor.text_vocab_size;types[is_motion]=1
    end=ids==processor.eom_id;holders=input_is_motion|output_is_motion
    ids[is_motion]-=processor.text_vocab_size
    target_labels=ids.clone().detach();target_labels[end]=-100;target_labels[holders]=-100
    with torch.no_grad():tokens=processor.motion_feats_to_tokens(vae,motion,lengths,modes=modes)
    count_in=input_is_motion.sum(dim=-1)//processor.holder_num_in_input
    count_out=output_is_motion.sum(dim=-1)//processor.motion_holder_repeat
    targets=[values[-count:] for values,count in zip(tokens,count_out) if count>0]
    return dict(type_ids=types,input_ids=ids,attention_mask=attention,motion_tokens=tokens,
        input_is_motion=input_is_motion,motion_num_in_input=count_in,output_is_motion=output_is_motion,
        output_hidden_states=True,return_diffusion_hidden=True),target_labels,targets


def language_loss(logits,labels,type_ids):
    total=0.
    for index,value in enumerate(logits):
        valid=type_ids==index
        if valid.sum()==0:continue
        if index==0:
            positions=valid[:,1:]
            if positions.sum()==0:loss=torch.tensor(0.).to(value)
            else:
                shifted=value[...,:-1,:][positions].contiguous()
                targets=labels[...,1:][positions].contiguous()
                loss=F.cross_entropy(shifted.view(-1,shifted.size(-1)),targets.view(-1),ignore_index=-100)
        else:loss=torch.tensor(0.).to(value)
        total=total+loss.nan_to_num_(0)
    return total
