import math
import os
import torch
from torch import nn
from einops import rearrange
from transformers import (
    GPT2LMHeadModel, 
    GPT2Model,
    AutoTokenizer
)
from .mot_example_gpt2_sepattn_gen import MoTGPT2LMHeadModel

import random
import numpy as np
def seed_setting(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    model_config = "deps/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left'

    model = GPT2LMHeadModel.from_pretrained(model_config).eval()
    model.transformer._attn_implementation = "sdpa"

    motion_holder_output_word = '<motion_latent_holder_output>'
    motion_holder_word = '<motion_latent_holder>'
    som_word, eom_word = '<start_of_motion>', '<end_of_motion>'
    mom_word, pom_word = '<masked_motion>', '<pad_motion>'
    new_special_tokens_text = [som_word, eom_word]
    new_special_tokens_mod = [mom_word, pom_word]
    new_tokens = [motion_holder_output_word, motion_holder_word]
    special_token_num = len(new_special_tokens_text) + len(new_special_tokens_mod)
    all_motion_token_num = special_token_num + len(new_tokens)

    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens_text})
    text_vocab_size = len(tokenizer)
    
    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens_mod})
    tokenizer.add_tokens(new_tokens)

    model2 = MoTGPT2LMHeadModel(model.config, motion_codebook_size=all_motion_token_num).eval()
    model2.generation_config = model.generation_config
    # new_s = dict()
    # for k, p in s.items():
    #     if 'lm.language_model' in k: 
    #         name = k.split('lm.language_model.')[-1]
    #         name = name.replace("attn.c_attn", "c_attn.fn.1")
    #         name = name.replace("attn.c_proj", "c_proj.fn.1")
    #         name = name.replace("mlp", "mlp.fn.1")
    #         name = name.replace("ln_f", "ln_f.fn.1")
    #         name = name.replace("ln_1", "ln_1.fn.1")
    #         name = name.replace("ln_2", "ln_2.fn.1")
    #         new_s[name]=p
    state_dict = model.state_dict()
    new_state_dict = dict()
    for name, param in state_dict.items():
        name = name.replace("attn.c_attn", "c_attn.fn.0")
        name = name.replace("attn.c_proj", "c_proj.fn.0")
        name = name.replace("mlp", "mlp.fn.0")
        name = name.replace("ln_f", "ln_f.fn.0")
        name = name.replace("ln_1", "ln_1.fn.0")
        name = name.replace("ln_2", "ln_2.fn.0")
        new_state_dict[name] = param

    # torch.save(new_state_dict, 'deps/mot-gpt2-medium/model_state_dict.pth')
    # new_state_dict = torch.load('deps/mot-gpt2/model_state_dict.pth')
    msg = model2.load_state_dict(new_state_dict, strict=False)
    
    model2.config.mot_lm_dim = model2.config.motion_vocab_size
    model2.config.text_vocab_size = text_vocab_size
    model2.set_modality_info(tokenizer)
    model2.resize_token_embeddings(text_vocab_size+2)

    inputs = tokenizer(["How are you today"], return_tensors="pt")
    _pre = True
    if _pre:
        # inputs1 = tokenizer([" A A B How are you today "], return_tensors="pt")
        inputs1 = tokenizer([" A A AHow are you today"],
                       padding='max_length',
                       return_tensors="pt", 
                       max_length=192,
                       truncation=True,
                       return_attention_mask=True,
                       add_special_tokens=True,)
        # [317,  317, 317, 2437,  389,  345, 1909])
    else:
        inputs1 = tokenizer(["How are you today A A A"], return_tensors="pt")
    
    # inputs2 = tokenizer(["translate: Hello"], return_tensors="pt")
    # type_ids = torch.zeros_like(inputs2.input_ids)
    # position_ids = torch.arange(inputs2.input_ids.shape[-1]).unsqueeze(0).long()

    generation_config = dict(
        max_new_tokens=20,
        do_sample=True,
        # use_cache=False,
        # top_k=1,
        # top_p=0.0001,
        num_return_sequences=1,
        # use_cache=False,
        pad_token_id=50256,
        # no_repeat_ngram_size=4,
    )

    input_ids = inputs1.input_ids
    inputs_embeds = model2.transformer.wte(inputs1.input_ids)
    type_ids = torch.zeros_like(inputs1.input_ids)
    position_ids = torch.arange(inputs1.input_ids.shape[-1]).unsqueeze(0).long()

    type_ids1 = type_ids.clone().detach()
    attention_mask1=inputs1.attention_mask.clone().detach()
    position_ids1 = position_ids.clone().detach()
    if _pre:
        type_ids1[...,:3] = 1
        attention_mask1[...,:3]=0
        position_ids1[...,3:] =  position_ids1[...,3:]- 3
    else:
        type_ids1[...,-4:-1] = 1
        attention_mask1[...,-4:-1]=0
        position_ids1[...,-4:-1] =  position_ids1[...,-4:-1]- position_ids1[...,-3]
        position_ids1[...,-1] = position_ids1[...,-5] +1


    torch.manual_seed(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs1 = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, 
                            **generation_config)
    print('inputs', tokenizer.batch_decode(outputs1, skip_special_tokens=True))
    
    torch.manual_seed(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs2 = model2.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, 
                            **generation_config)
    print('mine inputs', tokenizer.batch_decode(outputs2, skip_special_tokens=True))

    torch.manual_seed(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs1 = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=inputs1.attention_mask, 
                            **generation_config)
    print('inputs1', tokenizer.batch_decode(outputs1, skip_special_tokens=True))

    seed_setting(0)
    # inputs1.attention_mask[...,:3] = 0
    outputs1 = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask1, 
                            #   do_sample=False, use_cache=True,  )
                            **generation_config)
    print('attention_mask1', tokenizer.batch_decode(outputs1, skip_special_tokens=True))

    torch.manual_seed(0)
    # input_ids=input_ids
    outputs2 = model2.generate(input_ids=input_ids, attention_mask=inputs1.attention_mask,
                            type_ids=type_ids, # position_ids=position_ids, 
                            **generation_config)
    print('mine inputs1', tokenizer.batch_decode(outputs2, skip_special_tokens=True))

    torch.manual_seed(0)
    seed_setting(0)
    outputs2 = model2.generate(input_ids=input_ids, attention_mask=attention_mask1, 
                            type_ids=type_ids, # position_ids=position_ids, 
                            # do_sample=False, use_cache=True, )
                            **generation_config)
    print('mine attention_mask1', tokenizer.batch_decode(outputs2, skip_special_tokens=True))
    