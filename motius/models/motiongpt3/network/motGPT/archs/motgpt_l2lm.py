import os
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb


class MotionUndHead(nn.Module):
    def __init__(self, input_dim, output_dim, projector_type='linear', depth=1, **kwargs):
        super().__init__()
        if projector_type == "identity":
            modules = nn.Identity()

        elif projector_type == "linear":
            modules = nn.Linear(input_dim, output_dim)

        elif projector_type == "mlp_gelu":
            mlp_depth = depth
            modules = [nn.Linear(input_dim, output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
            modules = nn.Sequential(*modules)
        
        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")
        
        self.layers = modules

    def forward(self, motion_tokens):
        if len(motion_tokens.shape) == 3:
            bs, latent_dim, latent_channel = motion_tokens.shape
            motion_tokens = motion_tokens.permute(1,0,2)
            motion_tokens = motion_tokens.reshape(bs, -1)
        motion_embedding = self.layers(motion_tokens)
        return motion_embedding

def mask_grad(grad, allowed_ids):
    mask = torch.zeros_like(grad)
    mask[allowed_ids] = 1
    return grad * mask


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 192,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        vae_latent_channels: int = 256,
        vae_latent_size=None,
        # lm
        motion_holder_repeat = 4,
        holder_num_in_input = 4,
        motion_holder_seq_mode = 'withse',
        with_hid_norm = True,
        with_vae_latent_norm = True,
        # # diffloss
        # diffhead=None,
        # diffusion_batch_mul=4,
        # guidance_scale=1.0,
        # guidance_uncondp=0.1,
        # predict_epsilon=True,
        # fake_latent_mode=False,
        # mot arch
        mot_factor=1.0,
        attention_mode='all',        
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        self.with_hid_norm = with_hid_norm
        self.with_vae_latent_norm = with_vae_latent_norm

        self.motion_holder_output_word = '<motion_latent_holder_output>'
        self.motion_holder_word = '<motion_latent_holder>'
        self.som_word, self.eom_word = '<start_of_motion>', '<end_of_motion>'
        self.mom_word, self.pom_word = '<masked_motion>', '<pad_motion>'
        new_special_tokens_text = [self.som_word, self.eom_word]
        new_special_tokens_mod = [self.mom_word, self.pom_word]
        new_tokens = [self.motion_holder_output_word, self.motion_holder_word]
        special_token_num = len(new_special_tokens_text) + len(new_special_tokens_mod)
        all_motion_token_num = special_token_num + len(new_tokens)

        self.motion_holder_repeat = motion_holder_repeat
        self.holder_num_in_input = holder_num_in_input
         
        motion_holder_seq_mode = 'withse'
        if motion_holder_seq_mode == 'withse':
        # self.motion_holder_seq = '<motion_latent_holder>'*self.motion_holder_repeat
            self.output_motion_holder_seq = '<start_of_motion>'+self.motion_holder_output_word*self.motion_holder_repeat+'<end_of_motion>'
            self.input_motion_holder_seq = '<start_of_motion>'+self.motion_holder_word*self.holder_num_in_input+'<end_of_motion>'
        elif motion_holder_seq_mode == 'alone':
            self.output_motion_holder_seq = self.motion_holder_output_word*self.motion_holder_repeat
            self.input_motion_holder_seq = self.motion_holder_word*self.holder_num_in_input

        if model_type == "t5":
            assert False
            # lm = T5ForConditionalGeneration.from_pretrained(model_path)
            from transformers.models.t5.modeling_t5 import T5Config
            from transformers.generation.configuration_utils import GenerationConfig
            from motius.models.motiongpt3.network.mot_code.mot_example_t5 import MoTT5ForConditionalGeneration
            mconfig = T5Config.from_pretrained(f'{model_path}/config.json')
            # import json
            # with open('deps/mot-t5-base/generation_config.json', 'r') as file:
            #     gconfig = GenerationConfig(json.load(file))
            language_model = MoTT5ForConditionalGeneration(mconfig, motion_codebook_size=all_motion_token_num,
                                                           mot_factor=mot_factor, attention_mode=attention_mode
                                                           )
            # state_dict = torch.load(f'{model_path}/model_state_dict.pth')
            # # state_dict = torch.load('deps/mot-t5-base/mot_model_state_dict.pth')
           
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            # self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            from transformers.models.gpt2.modeling_gpt2 import GPT2Config
            from motius.models.motiongpt3.network.mot_code.mot_example_gpt2 import MoTGPT2LMHeadModel
            mconfig = GPT2Config.from_pretrained(f'{model_path}/config.json')
            language_model = MoTGPT2LMHeadModel(
                    mconfig, motion_codebook_size=all_motion_token_num, 
                    mot_factor=mot_factor, attention_mode=attention_mode
                )
            language_model.transformer._attn_implementation = "sdpa"
            # state_dict = torch.load(f'{model_path}/model_state_dict.pth')
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")


        # Instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        len_raw_tokenizer = len(self.tokenizer)
        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens_text})
        self.text_vocab_size = len(self.tokenizer)
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens_mod})
        self.tokenizer.add_tokens(new_tokens)

        self.som_id, self.eom_id =  self.tokenizer.convert_tokens_to_ids([self.som_word, self.eom_word])
        # self.mom_id, self.pom_id =  self.tokenizer.convert_tokens_to_ids([self.mom_word, self.pom_word])
        self.motion_holder_id_out, self.motion_holder_id = self.tokenizer.convert_tokens_to_ids([self.motion_holder_output_word, self.motion_holder_word])
        
        # load state_dict from pth
        state_dict = torch.load(f'{model_path}/model_state_dict.pth')
        new_state_dict = state_dict.copy()
        if 'encoder.embed_tokens.weight' in new_state_dict: # model_type == "t5"
            new_state_dict.pop('encoder.embed_tokens.weight')
            new_state_dict.pop('decoder.embed_tokens.weight')

        msg = language_model.load_state_dict(new_state_dict, strict=False)
        if len(msg.unexpected_keys)>0:
            print('unexpected_keys keys:', msg.unexpected_keys)
            exit()
        # print('missing keys:', msg.missing_keys)
        self.language_model = language_model
        self.language_model.train()
        
        text_embeddings_num = self.text_vocab_size
        self.language_model.resize_token_embeddings(text_embeddings_num)
        
        # self.language_model.config.mot_lm_dim = self.language_model.config.motion_vocab_size
        # print('mot_lm_dim', self.language_model.config.mot_lm_dim)
        self.language_model.set_modality_info(self.tokenizer)
        self.mod2id = self.language_model.mod2id.copy()
        self.mod_id = self.mod2id['motion']
        # self.language_model.modality_infos[1].post_processor
        # self.language_model.init_mod_token_embeddings(mod_id=1)
        
        mot_trained = True
        text_trained = 'finetune' in stage
        for n, p in self.language_model.named_parameters():
            if n in state_dict.keys() or 'fn.0' in n:
                p.requires_grad = text_trained
            else:
                p.requires_grad = mot_trained
                # print(n, end=' ')
                
        if not text_trained and text_embeddings_num > len_raw_tokenizer:
            self.language_model.pre_processors[0].weight.requires_grad = True
            self.language_model.post_processors[0].weight.requires_grad = True
            allow_ids = list(range(len_raw_tokenizer, text_embeddings_num))
            self.language_model.pre_processors[0].weight.register_hook(
                lambda grad: mask_grad(grad, allow_ids)
            )
            self.language_model.post_processors[0].weight.register_hook(
                lambda grad: mask_grad(grad, allow_ids)
            )

        # adaptor from vae latent to llm latent
        self.llm_decoder_embed_dim = self.language_model.mot_embed_dim  # 768
        self.motion_und_head = MotionUndHead(vae_latent_channels*2, self.llm_decoder_embed_dim*self.holder_num_in_input, projector_type='linear')

        self.motion_gen_head = MotionUndHead(self.llm_decoder_embed_dim*self.motion_holder_repeat, vae_latent_channels, projector_type='linear')
        self.motion_loss = nn.MSELoss()

        # # Lora
        # if lora:
        #     from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
        #     from peft.utils.other import fsdp_auto_wrap_policy
        #     peft_config = LoraConfig(
        #         bias="none",
        #         task_type="CAUSAL_LM",
        #         #  inference_mode=False,
        #         r=8,
        #         lora_alpha=16,
        #         lora_dropout=0.05)
        #     self.language_model = get_peft_model(self.language_model,
        #                                          peft_config)

    def forward(self, texts: List[str], motion_feats: Tensor, motion_encode_net,
                lengths: List[int], tasks: dict,
                output_hidden_states=False
                ):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_feats, motion_encode_net, lengths, tasks, 
                                       output_hidden_states=output_hidden_states, 
                                       )
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_feats, motion_encode_net, lengths, tasks, 
                                    output_hidden_states=output_hidden_states, 
                                    )
        else:
            raise NotImplementedError("Only conditional_multitask supported")
    
    def sample_tokens(self, outputs, device, temperature=1.0, cfg=1.0, vae_mean_std_inv=None):
        # todo: allow more motion sequence in output 
        if isinstance(outputs, List):
            output_to_sample = []
            motion_mask = torch.zeros(len(outputs))
            fake_embedding = torch.zeros((self.motion_holder_reapeat, self.llm_decoder_embed_dim))
            for i, output in enumerate(outputs):
                if output is not None:  # [m_num,768]
                    output_to_sample.append(output[:,:])
                else:
                    output_to_sample.append(fake_embedding)
                    motion_mask[i] = 1
            output_to_sample = torch.stack(output_to_sample, dim=0).to(device)  #[bs,10,768]
        else:
            motion_mask = torch.zeros(len(outputs))
            output_to_sample = outputs.to(device)  #[bs,10,768]
        sampled_token_latents = self.motion_gen_head(output_to_sample.reshape(len(motion_mask), -1))

        if self.with_vae_latent_norm:
            sampled_token_latents = sampled_token_latents/vae_mean_std_inv
        return sampled_token_latents, motion_mask

    def forward_encdec(
        self,
        texts: List[str],
        motion_feats: Tensor,
        motion_encode_net, 
        lengths: List[int],
        tasks: dict,
        # output_hidden_states=False,
    ):

        # # Tensor to string
        # motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            inputs = texts
            # outputs = texts
        elif condition == 'motion':
            inputs = self.input_motion_holder_seq
            outputs = self.output_motion_holder_seq
            modes = 'motion'
        else:
            # inputs, outputs = self.template_fulfill(tasks, lengths,
            #                                         motion_strings, texts)
            inputs, outputs, modes = self.template_fulfill(tasks, lengths, texts)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(motion_feats.device)
        source_attention_mask = source_encoding.attention_mask.to(motion_feats.device)
        input_is_motion = source_input_ids==self.motion_holder_id
        batch_size, expandend_input_length = source_input_ids.shape

        if condition in ['text', 'motion']:
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids, target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids, input_ids_sentinel)
        else:
            if '2t' in ' '.join(modes):
                max_len = self.max_length
            else:
                max_len = self.motion_holder_repeat +3
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=max_len,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_feats.device)
            lables_attention_mask = target_inputs.attention_mask.to(motion_feats.device)
        output_is_motion = labels_input_ids==self.motion_holder_id_out
            
        type_ids = torch.zeros_like(source_input_ids).to(motion_feats.device)
        input_type_ind = source_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size

        output_type_ids = torch.zeros_like(labels_input_ids).to(motion_feats.device)
        output_type_ind = labels_input_ids>=self.text_vocab_size
        output_type_ids[output_type_ind] = 1
        ignore_ind = labels_input_ids == 0
        eom_ignore = labels_input_ids==self.eom_id
        # output_type_ids[labels_input_ids == -100] = -1
        labels_input_ids[output_type_ind] -= self.text_vocab_size
        decoder_input_ids = self.language_model._shift_right(labels_input_ids)
        
        labels_input_ids[ignore_ind|eom_ignore] = -100
        # labels_input_ids[output_is_motion] = self.mot_pad_id  # add in v2
        
        with torch.no_grad():
            motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes=modes)

        inputs_embeds = self.language_model.get_embeddings_from_ids(source_input_ids, type_ids)
        # decoder_inputs_embeds = self.language_model.get_embeddings_from_ids(decoder_input_ids)

        # motion_tokens: List [bs*[k,1,256]]
        # dec_input_is_motion = decoder_input_ids==self.motion_holder_id_out
        motion_num_in_input = input_is_motion.sum(dim=-1)//self.holder_num_in_input
        motion_num_in_label = output_is_motion.sum(dim=-1)//self.motion_holder_repeat
        # print(modes)
        # print('self.motion_holder_id', self.motion_holder_id)
        # print(motion_num_in_input+motion_num_in_label)
        # print(motion_num_in_label)
        for i in range(batch_size):
            if motion_tokens[i] is None: # motion_tokens[i]: [motion_num,1,256]
                continue
            motion_embeddings = self.motion_und_head(motion_tokens[i]) # [motion_num, 768]
            motion_embs = motion_embeddings[:motion_num_in_input[i]].split(self.llm_decoder_embed_dim,dim=-1)
            inputs_embeds[self.mod_id][i, input_is_motion[i], :] = torch.cat(motion_embs, dim=0)
            # decoder_inputs_embeds[i, dec_input_is_motion[i], :] = motion_embeddings[motion_num_in_input[i]:]
        ll = [m[-num:] for m, num in zip(motion_tokens, motion_num_in_label) if num>0]
        del motion_tokens, motion_feats

        outputs = self.language_model(
            # input_ids=source_input_ids,
            type_ids=type_ids,
            decoder_type_ids=output_type_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=source_attention_mask
                if condition == 'supervised' else None,
            labels=labels_input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=lables_attention_mask
                if condition == 'supervised' else None,
            # decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=output_hidden_states, 
        )
        hidden = outputs.decoder_hidden_states[-1]  # [bs, labels_seq_len, emb_dim(768)] 
        # if motion_in_label.sum()>0:
        #     ll = []
        #     hidden_to_diff = []
        #     for i,m in enumerate(motion_in_label):
        #         if not m: continue
        #         ll.append(motion_tokens[i][-1:])
        #         hidden_to_diff.append(hidden[i:i+1, output_is_motion[i], :])
        # else:
        #     outputs = self.language_model(
        #         type_ids=type_ids,
        #         decoder_type_ids=output_type_ids,
        #         input_ids=source_input_ids,
        #         attention_mask=source_attention_mask
        #             if condition == 'supervised' else None,
        #         labels=labels_input_ids,
        #         decoder_attention_mask=lables_attention_mask
        #             if condition == 'supervised' else None,
        #         output_hidden_states=output_hidden_states, 
        #     )
        #     ll = [m[-1:] for m in motion_tokens]
        #     hidden = outputs.decoder_hidden_states[-1][self.mod_id]  # [bs, labels_seq_len, emb_dim(768)] 
        #     # hidden_to_diff = hidden[:, :self.motion_holder_repeat,:]

        try:
        # if len(ll)>0:
            ll = torch.cat(ll, dim=0)  # bs,labels_seq_len,256
            hidden_to_diff = [hidden[i:i+1,m, :] for i,m in enumerate(output_is_motion) if m.sum()>0]
            hidden_to_diff = torch.cat(hidden_to_diff, dim=0)  # bs, motion_holder_reapeat, 768
            # hidden_to_diff = self.norm_layer(hidden_to_diff)
            # hidden_to_diff = hidden_to_diff + self.diffusion_pos_embed_learned
            # print(hidden_to_diff.shape, ll.shape)
            outputs.diff_loss = self.forward_diff_loss(z=hidden_to_diff, target=ll)
            # outputs.diff_loss, xstart = self.forward_diff_loss(z=hidden_to_diff, target=ll)
            # hidden[0][output_is_motion[0], :].shape
            del hidden_to_diff, ll
        except:
            outputs.diff_loss = torch.tensor(0.)
        
        # outputs.loss.nan_to_num_(0)
        # outputs.diff_loss.nan_to_num_(0)
        torch.cuda.empty_cache()
        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_feats: Tensor,
        motion_encode_net, 
        lengths: List[int],
        tasks: dict,
        output_hidden_states=True,
    ):
        # assert False
        self.tokenizer.padding_side = "right"
        # self.tokenizer.padding_side = "left"

        with torch.no_grad():
            # Supervised or unsupervised
            condition = random.choice(
                ['supervised', 'supervised', 'supervised'])
                # ['text', 'motion', 'supervised', 'supervised', 'supervised'])

            if condition == 'text':
                modes = 'text'
                labels = texts
            elif condition == 'motion':
                modes = 'motion'
                labels = self.input_motion_holder_seq+' \n '+self.output_motion_holder_seq + self.tokenizer.eos_token
            else:
                inputs, outputs, modes = self.template_fulfill(tasks, lengths, texts)
                # inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)
                labels = []
                for i in range(len(inputs)):
                    labels.append(inputs[i] + ' \n ' + outputs[i] +
                                self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_feats.device)
        lables_attention_mask = inputs.attention_mask.to(motion_feats.device)
        input_is_motion = labels_input_ids==self.motion_holder_id
        output_is_motion = labels_input_ids==self.motion_holder_id_out

        # type_ids = torch.ones_like(labels_input_ids).to(motion_tokens.device)
        type_ids = torch.zeros_like(labels_input_ids).to(motion_feats.device)
        input_type_ind = labels_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        eom_ignore = labels_input_ids==self.eom_id
        labels_input_ids[input_type_ind] -= self.text_vocab_size

        inputs_embeds = self.language_model.get_embeddings_from_ids(labels_input_ids, type_ids)
        motion_num_in_input = input_is_motion.sum(dim=-1)//self.holder_num_in_input
        motion_num_in_label = output_is_motion.sum(dim=-1)//self.motion_holder_repeat

        labels = labels_input_ids.clone().detach()
        labels[eom_ignore] = -100
        # labels[output_is_motion|input_is_motion] = -100
        
        batch_size, expandend_input_length = labels_input_ids.shape
        with torch.no_grad():
            motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes=modes)
        
        for i in range(batch_size):
            if (motion_tokens[i] is None) or (motion_num_in_input[i]<1): # motion_tokens[i]: [motion_num,1,256]
                continue
            motion_embeddings = self.motion_und_head(motion_tokens[i]) # [motion_num, 768]
            motion_embs = motion_embeddings[:motion_num_in_input[i]].split(self.llm_decoder_embed_dim,dim=-1) # tuple(4, [1, 768])
            inputs_embeds[self.mod_id][i, input_is_motion[i], :] = torch.cat(motion_embs, dim=0)
            # inputs_embeds[self.mod_id][i, input_is_motion[i], :] = motion_embeddings[:motion_num_in_input[i]]
        ll = [m[-num:] for m, num in zip(motion_tokens, motion_num_in_label) if num>0]
        del motion_tokens, motion_feats
            
        outputs = self.language_model(
            type_ids=type_ids,
            inputs_embeds=inputs_embeds,
            # input_ids=labels_input_ids,
            attention_mask=lables_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states, 
            )
        
        try:
        # if len(ll)>0:
            ll = torch.cat(ll, dim=0)  # bs, labels_seq_len,256
            hidden = outputs.hidden_states[-1][self.mod_id][:, :-1,:]  # [bs, labels_seq_len, emb_dim(768)] 
            # target size (torch.Size([32, 1, 256]).
            hidden_to_l2 = [hidden[i:i+1,m, :] for i,m in enumerate(output_is_motion[:, 1:]) if m.sum()>0]
            hidden_to_l2 = torch.cat(hidden_to_l2).view(ll.shape[0], -1)
            # hidden_to_l2 = torch.cat(hidden_to_l2).reshape(
            #     ll.shape[0], -1, self.llm_decoder_embed_dim*self.holder_num_in_input)
            # hidden_to_l2.shape: torch.Size([1, 1, 3072])
            # ll.shape: torch.Size([1, 1, 256])
            pred_motion = self.motion_gen_head(hidden_to_l2).view(ll.shape)
            outputs.diff_loss = self.motion_loss(pred_motion, ll)
        except:
            outputs.diff_loss = torch.tensor(0.)
        # if len(ll)>0 :
        #     ll = torch.cat(ll, dim=0)  # bs,labels_seq_len,256
        #     hidden_to_diff = [hidden[i:i+1,m, :] for i,m in enumerate(output_is_motion) if m.sum()>0]
        #     hidden_to_diff = torch.cat(hidden_to_diff)
        #     hidden_to_diff = self.norm_layer(hidden_to_diff)
        #     hidden_to_diff = hidden_to_diff + self.diffusion_pos_embed_learned
        #     # print(hidden_to_diff.shape, ll.shape)
        #     outputs.diff_loss = self.forward_diff_loss(z=hidden_to_diff, target=ll)
        #     # outputs.diff_loss, xstart = self.forward_diff_loss(z=hidden_to_diff, target=ll)
        #     # hidden[0][output_is_motion[0], :].shape
        #     del hidden_to_diff, ll
        # else:
        #     outputs.diff_loss = torch.tensor(0.)
        # outputs.loss.nan_to_num_(0)
        # outputs.diff_loss.nan_to_num_(0)
        torch.cuda.empty_cache()
        return outputs

    def generate_direct(self,
                        texts: List[str],
                        motion_tokens=None,  # motion_tokens: List [bs*[k,1,256]]
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None,
                        output_hidden_states=False,
                        gen_mode:Union[str, List[str]] = 'text'):

        # Device
        self.device = self.language_model.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + " \n " for text in texts]
            self.tokenizer.padding_side = "left"


        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        input_is_motion = source_input_ids==self.motion_holder_id

        type_ids = torch.zeros_like(source_input_ids)
        input_type_ind = source_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size

        inputs_embeds = self.language_model.get_embeddings_from_ids(source_input_ids, type_ids)  # [bs,256,768]
                
        if motion_tokens is not None: # motion_num,1,256
            motion_num_in_input = input_is_motion.sum(dim=-1)//self.holder_num_in_input
            for i, source_input_id in enumerate(source_input_ids):
                motion_embeddings = self.motion_und_head(motion_tokens[i])
                motion_embs = motion_embeddings[:motion_num_in_input[i]].split(self.llm_decoder_embed_dim,dim=-1)
                inputs_embeds[self.mod_id][i, input_is_motion[i], :] = torch.cat(motion_embs, dim=0)
                # inputs_embeds[self.mod_id][i, input_is_motion[i], :] = motion_embeddings[:motion_num_in_input[i]]
        else:
            assert (~input_is_motion).all(), 'motion holder in input text, wihout provided motion tokens'
        del motion_tokens

        mid = self.mod2id[gen_mode]
        som_id = self.language_model.modality_infos[mid].som_id
        eom_id = self.language_model.modality_infos[mid].eom_id + self.text_vocab_size
        # decoder_input_ids = torch.full((source_input_ids.shape[0], 1), som_id).to(source_input_ids)
        # decoder_type_ids = torch.full((source_input_ids.shape[0], 1), mid).to(source_input_ids)

        with torch.no_grad():
            if self.lm_type == 'encdec':
                outputs = self.language_model.generate(
                    type_ids=type_ids,
                    # input_ids=source_input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=source_attention_mask,
                    # decoder_input_ids=decoder_input_ids,
                    # decoder_type_ids=decoder_type_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    bad_words_ids=bad_words_ids,
                    return_dict_in_generate=True,
                    output_hidden_states=output_hidden_states,
                    mode=gen_mode,
                )
            elif self.lm_type == 'dec':
                # # assert False
                # if gen_mode=='motion':
                #     source_input_ids = torch.cat([source_input_ids, decoder_input_ids], -1)
                #     attn = source_encoding.attention_mask
                #     source_encoding.attention_mask = torch.cat([attn, torch.ones_like(attn[:,-1:])], -1)
                #     type_ids = torch.cat(
                #         [type_ids, torch.full(type_ids[:,-1:].shape, mid, dtype=torch.long, device=self.device)], -1)

                outputs = self.language_model.generate(
                    type_ids=type_ids,
                    # input_ids=source_input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=source_attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    max_new_tokens=max_length,
                    return_dict_in_generate=True,
                    output_hidden_states=output_hidden_states,
                    mode=gen_mode,
                    repetition_penalty=1.1,
                )
                self.tokenizer.padding_side = 'left'

        output_ids = outputs.sequences
        outputs1 = torch.cat([output_ids, torch.full((*output_ids.shape[:-1],1), eom_id, device=self.device, dtype=output_ids.dtype)], dim=-1)
        outputs_string = self.tokenizer.batch_decode(outputs1, skip_special_tokens=False)

        motion_string, cleaned_text = self.clean_text_string(outputs_string)
        # outputs_tokens, cleaned_text = self.motion_string_to_token(
        #     outputs_string)
        del outputs
        motion_latents = []

        return motion_latents, cleaned_text

    def generate_direct_motion(self,
                        texts: List[str],
                        motion_tokens=None,  # motion_tokens: List [bs*[k,1,256]]
                        # # fixed_motion_length: int = 8,
                        # num_beams: int = 1,
                        # do_sample: bool = True,
                        # bad_words_ids: List[int] = None,
                        ):
        
        # Device
        self.device = self.language_model.device
        bsz = len(texts)
        # Tokenize
        if self.lm_type == 'dec':
            # self.tokenizer.padding_side = 'left'
            texts = [text + " \n " + self.output_motion_holder_seq +
                                self.tokenizer.eos_token for text in texts]

        source_encoding = self.tokenizer(texts,
                                            padding='max_length',
                                            max_length=self.max_length,
                                            truncation=True,
                                            return_attention_mask=True,
                                            add_special_tokens=True,
                                            return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        input_is_motion = source_input_ids==self.motion_holder_id
        output_is_motion = source_input_ids==self.motion_holder_id_out

        type_ids = torch.zeros_like(source_input_ids)
        input_type_ind = source_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size

        inputs_embeds = self.language_model.get_embeddings_from_ids(source_input_ids, type_ids)  # [bs,256,768]
            
        if motion_tokens is not None: # motion_num,1,256
            motion_num_in_input = input_is_motion.sum(dim=0)//self.holder_num_in_input
            for i, source_input_id in enumerate(source_input_ids):
                motion_embeddings = self.motion_und_head(motion_tokens[i])
                motion_embs = motion_embeddings[:motion_num_in_input[i]].split(self.llm_decoder_embed_dim,dim=-1)
                inputs_embeds[self.mod_id][i, input_is_motion[i], :] = torch.cat(motion_embs, dim=0)
                # inputs_embeds[self.mod_id][i, input_is_motion[i], :] = motion_embeddings[:motion_num_in_input[i]]
        else:
            assert (~input_is_motion).all(), 'motion holder in input text, wihout provided motion tokens'
        del motion_tokens
        if self.lm_type == 'encdec':
            target_inputs = self.tokenizer([self.output_motion_holder_seq],
                                           padding='max_length',
                                           max_length=self.motion_holder_repeat+3,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")
            
            labels_input_ids = target_inputs.input_ids.to(inputs_embeds.device)
            lables_attention_mask = target_inputs.attention_mask.to(inputs_embeds.device)
            output_is_motion = labels_input_ids==self.motion_holder_id_out

            output_type_ids = torch.zeros_like(labels_input_ids).to(inputs_embeds.device)
            output_type_ind = labels_input_ids>=self.text_vocab_size
            output_type_ids[output_type_ind] = 1
            ignore_ind = labels_input_ids == 0
            # output_type_ids[labels_input_ids == -100] = -1
            labels_input_ids[output_type_ind] -= self.text_vocab_size
            decoder_input_ids = self.language_model._shift_right(labels_input_ids)
            
            labels_input_ids[ignore_ind] = -100
            outputs = self.language_model(
                # input_ids=source_input_ids,
                type_ids=type_ids,
                decoder_type_ids=output_type_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=source_attention_mask,
                # labels=labels_input_ids.repeat(bsz, 1),
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=lables_attention_mask.repeat(bsz, 1),
                output_hidden_states=True, 
            )
            hidden = outputs.decoder_hidden_states[-1][self.mod_id] # bs, tok_max_len, lat_dim(768)
            latents = torch.stack([h[output_is_motion[i], :] for i,h in enumerate(hidden)]) # bs, motion_holder_repeat, lat_dim(768)
        elif self.lm_type == 'dec':
            # motion_num_in_label = output_is_motion.sum(dim=-1)//self.motion_holder_repeat
            labels = source_input_ids.clone().detach()
            # labels[output_is_motion] = self.mot_pad_id
            # labels[input_is_motion] = self.mot_pad_id
            outputs = self.language_model(
                type_ids=type_ids,
                # input_ids=source_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=source_attention_mask,
                # labels=labels,
                output_hidden_states=True, 
                )
            hidden = outputs.hidden_states[-1][self.mod_id][:,:-1,:]  # [bs, labels_seq_len, emb_dim(768)] 
            output_is_motion = output_is_motion[:,1:]
            # hidden = outputs.hidden_states[-1][self.mod_id]  # [bs, labels_seq_len, emb_dim(768)] 
            latents = torch.stack([h[output_is_motion[i], :] for i,h in enumerate(hidden)]) # bs, motion_holder_repeat, lat_dim(768)

        del outputs
        # latents.mean(),  latents.std()
        # # (tensor(0.2327, device='cuda:0'), tensor(6.2979, device='cuda:0')
        # latents = self.norm_layer(latents)
        # latents = latents + self.diffusion_pos_embed_learned
        return latents

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_feats: Optional[Tensor] = None,
                             motion_encode_net=None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None
                            #  output_hidden_states=False
                             ):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
                # motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{
                            'class': 't2m',
                            'input':
                            ['Generate motion: <Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)

                    lengths = [0] * len(texts)
                else:
                    assert False, 'not Implemented'
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
                motion_tokens = None
                    
            elif task == "pred":
                # assert False, 'not Implemented'
                assert motion_feats is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'class': 'pred',
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                with torch.no_grad():
                    # motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='motion')
                    motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='pred')

            elif task == "inbetween":
                # assert False, 'not Implemented'
                assert motion_feats is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'class': 'inbetween',
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                
                with torch.no_grad():
                    # motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='motion')
                    motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='inbetween')
                # # motion_tokens: List [bs*[k,1,256]]

            inputs, outputs, modes = self.template_fulfill(tasks, lengths, texts, stage)

            outputs_tokens = self.generate_direct_motion(
                inputs,
                motion_tokens=motion_tokens,
                # num_beams=1,
                # do_sample=True,
            )

            return outputs_tokens

        elif task in ["m2t", "t2t"]:
            if task == 'm2t':
                assert motion_feats is not None and lengths is not None

                with torch.no_grad():
                    motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='motion')

                if not with_len:
                    tasks = [{
                        'class': 'm2t',
                        'input': ['Generate text: <Motion_Placeholder>'],
                        'output': ['']
                    }] * len(lengths)
                else:
                    tasks = [{
                        'class': 'm2t',
                        'input': [
                            'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(lengths)
            elif task == 't2t':
                tasks = [{
                    'class': 't2t',
                    'input': ['<Caption_Placeholder>'],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs, modes = self.template_fulfill(tasks, lengths, texts)
            
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                motion_tokens=motion_tokens,
                max_length=40,
                num_beams=1,
                do_sample=False,
                gen_mode='text',
            )
            return cleaned_text

    def motion_feats_to_tokens(self, motion_encode_net, motion_feats: Tensor, lengths: List[int], 
                               modes='motion', motion_tokens=None):
        motion_tokens = []
        # motion_tokens_input = []
        bs, _ , feat_dim = motion_feats.shape
        if isinstance(modes, str):
            modes = [modes]*bs
        for i in range(bs):
            mode = modes[i]
            length = lengths[i]
            max_len = length
            motion_feat = motion_feats[i,:length,:]  # seq_length, feat_dim(256)
            motion_feats_to_encode = torch.zeros((3, length, feat_dim), requires_grad=False)
            if mode in ['motion', 't2m', 'm2t', 'l2m', 'n2m', 'm2l']:
                motion_feats_to_encode[0, :, :] = motion_feat
                lengths_input = [length]

            elif mode in ['predict', 'pred']:
                predict_head = int(length * self.predict_ratio)
                motion_feats_to_encode[0, :predict_head, :] = motion_feat[:predict_head, :]
                motion_feats_to_encode[1, :length-predict_head, :] = motion_feat[predict_head:, :]
                lengths_input = [predict_head, length-predict_head]
                max_len = max(predict_head, length-predict_head)
                
            elif mode in ['inbetween', 'inbetweening']:
                masked_head = int(length * self.inbetween_ratio)
                masked_tail = int(length * (1 - self.inbetween_ratio))
                motion_feats_to_encode[0, :masked_head, :] = motion_feat[:masked_head, :]
                motion_feats_to_encode[1, :length-masked_tail, :] = motion_feat[masked_tail:, :]
                motion_feats_to_encode[2, :, :] = motion_feat
                lengths_input = [masked_head, length-masked_tail, length]
            elif mode in ['t2t','n2t']:
                motion_tokens.append(None)
                # motion_tokens_input.append(None)
                continue
            else:
                assert False, mode
            motion_feats_to_encode = motion_feats_to_encode[:len(lengths_input), :max_len]
            # z, dist = motion_encode_net.encode(motion_feats_to_encode.to(motion_feats.device), lengths_input)

            dist = motion_encode_net.encode_dist(motion_feats_to_encode.to(motion_feats.device), lengths_input)
            z, _ = motion_encode_net.encode_dist2z(dist)
            
            if self.with_vae_latent_norm:
                motion_tokens.append(z.permute(1,0,2).mul_(motion_encode_net.mean_std_inv))
            else:
                motion_tokens.append(z.permute(1,0,2))
            # motion_dist.append(dist)

        return motion_tokens # List[List[tensor]([bs, 1, 256])]
    
    def clean_text_string(self, motion_string: List[str]):
        output_string = []
        for i in range(len(motion_string)):
            output_string.append(motion_string[i].replace(
                self.motion_holder_output_word, '<Motion_Placeholder>').replace(
                    self.tokenizer.pad_token, ''))
    #         try:
    #             last = out.index('\n')
    #             out = out[:last]
    #         except:
    #             pass
    #             # last = len(out)+1
            
    #         try:
    #             last = out.index('.')
    #             out = out[:last+2]
    #         except:
    #             pass
            
    #         try:
    #             start = out.index('\"')
    #             end = out[start+2:].index('\"')
    #             out = out[start+1:start+2+end]
    #         except:
    #             pass
    #         output_string.append(out)

        return None, output_string
    
    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str):

        seconds = math.floor(length / self.framerate)
        motion_masked = motion_string + '<motion_masked>' + motion_string
        
        if random.random() < self.quota_ratio and ('\"' not in text):
            text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>', motion_string).replace(
                '<Frame_Placeholder>', f'{length}').replace(
                    '<Second_Placeholder>', '%.1f' % seconds).replace(
                        '<Motion_Placeholder_s1>', motion_string).replace(
                            '<Motion_Placeholder_s2>', motion_string).replace(
                                '<Motion_Placeholder_Masked>', motion_masked)
        if 'Placeholder' in prompt:
            startIndex = prompt.index('Placeholder')
            assert False, 'found unsupported PlaceHolder' + prompt[startIndex-8:startIndex+20]

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                        #  motion_strings,
                         texts,
                         stage='test'):
        inputs = []
        outputs = []
        modes = []

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length, self.input_motion_holder_seq, texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length, self.output_motion_holder_seq, texts[i]))
            modes.append(tasks[i]['class'])
            # 'predict' 'motion' 'inbetween' '2t'

        return inputs, outputs, modes

    # def get_middle_str(self, content, startStr, endStr):
    #     try:
    #         startIndex = content.index(startStr)
    #         if startIndex >= 0:
    #             startIndex += len(startStr)
    #         endIndex = content.index(endStr)
    #     except:
    #         return '<start_of_motion><motion_id_0><end_of_motion>'
    #         # return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'
    #     return startStr + content[startIndex:endIndex] + endStr

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
