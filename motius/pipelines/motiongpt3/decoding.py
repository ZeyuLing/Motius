"""MotionGPT3 multimodal task decoding and latent diffusion sampling."""
import torch,random,math
import numpy as np
from typing import List,Union,Optional
from torch import Tensor
from motius.motion.conditioning.motiongpt3 import MotionGPT3Processor
from .language_sampling import MotionGPT3LanguageSampler
from motius.diffusion.motiongpt3 import create_diffusion
from functools import partial

class MotionGPT3Decoder(MotionGPT3Processor):
    def __init__(self,bundle):
        from transformers import AutoTokenizer
        super().__init__(AutoTokenizer.from_pretrained(bundle.tokenizer_path,legacy=True,local_files_only=True),bundle.options,bundle.device)
        self.bundle=bundle
        self.language_model=MotionGPT3LanguageSampler(bundle.unwrap_network(bundle.lm).language_model)
        self.mod2id=self.language_model.mod2id;self.mod_id=self.mod2id['motion'];self.mot_pad_id=self.language_model.modality_infos[self.mod_id].pad_id
        self.llm_decoder_embed_dim=self.language_model.mot_embed_dim
        self.with_hid_norm=self.options_with_hid_norm if hasattr(self,'options_with_hid_norm') else bundle.options['with_hid_norm']
        self.hidden_dim=self.llm_decoder_embed_dim if self.multi_hidden else self.llm_decoder_embed_dim*self.motion_holder_repeat
        self.diffusion=MotionGPT3DiffusionSampler(bundle)
    @property
    def motion_und_head(self):return self.bundle.unwrap_network(self.bundle.lm).motion_und_head
    @property
    def norm_layer(self):return self.bundle.unwrap_network(self.bundle.lm).norm_layer
    @property
    def diffusion_pos_embed_learned(self):return self.bundle.unwrap_network(self.bundle.lm).diffusion_pos_embed_learned
    @property
    def fake_latent(self):return self.bundle.unwrap_network(self.bundle.diffusion_head).fake_latent

    def generate_direct(self, texts: List[str], motion_tokens=None, max_length: int=256, num_beams: int=1, do_sample: bool=True, bad_words_ids: List[int]=None, output_hidden_states=False, gen_mode: Union[str, List[str]]='text'):
        self.device = self.language_model.device
        if self.lm_type == 'dec':
            texts = [text + ' \n' for text in texts]
            self.tokenizer.padding_side = 'left'
        source_encoding = self.tokenizer(texts, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        input_is_motion = source_input_ids == self.motion_holder_id
        type_ids = torch.zeros_like(source_input_ids)
        input_type_ind = source_input_ids >= self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size
        inputs_embeds = self.language_model.get_embeddings_from_ids(source_input_ids, type_ids)
        if motion_tokens is not None:
            motion_num_in_input = input_is_motion.sum(dim=-1) // self.holder_num_in_input
            for i, source_input_id in enumerate(source_input_ids):
                motion_embeddings = self.motion_und_head(motion_tokens[i])
                motion_embs = motion_embeddings[:motion_num_in_input[i]].split(self.llm_decoder_embed_dim, dim=-1)
                inputs_embeds[self.mod_id][i, input_is_motion[i], :] = torch.cat(motion_embs, dim=0)
        else:
            assert (~input_is_motion).all(), 'motion holder in input text, wihout provided motion tokens'
        mid = self.mod2id[gen_mode]
        som_id = self.language_model.modality_infos[mid].som_id + self.language_model.modality_infos[mid].token_id_start
        eom_id = self.language_model.modality_infos[mid].eom_id + self.language_model.modality_infos[mid].token_id_start
        with torch.no_grad():
            if self.lm_type == 'encdec':
                outputs = self.language_model.generate(type_ids=type_ids, inputs_embeds=inputs_embeds, attention_mask=source_attention_mask, max_length=max_length, num_beams=num_beams, do_sample=do_sample, bad_words_ids=bad_words_ids, return_dict_in_generate=True, output_hidden_states=output_hidden_states, mode=gen_mode)
            elif self.lm_type == 'dec':
                outputs = self.language_model.generate(type_ids=type_ids, inputs_embeds=inputs_embeds, attention_mask=source_attention_mask, pad_token_id=self.tokenizer.pad_token_id, do_sample=do_sample, bad_words_ids=bad_words_ids, max_new_tokens=max_length, return_dict_in_generate=True, output_hidden_states=output_hidden_states, mode=gen_mode, repetition_penalty=1.1)
        output_ids = outputs.sequences
        outputs1 = torch.cat([output_ids, torch.full((*output_ids.shape[:-1], 1), eom_id, device=self.device, dtype=output_ids.dtype)], dim=-1)
        outputs_string = self.tokenizer.batch_decode(outputs1, skip_special_tokens=False)
        motion_string, cleaned_text = self.clean_text_string(outputs_string)
        motion_latents = []
        return (motion_latents, cleaned_text)

    def generate_direct_motion(self, texts: List[str], motion_tokens=None):
        self.device = self.language_model.device
        bsz = len(texts)
        if self.lm_type == 'dec':
            texts = [text + ' \n ' + self.output_motion_holder_seq + self.tokenizer.eos_token for text in texts]
        source_encoding = self.tokenizer(texts, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        input_is_motion = source_input_ids == self.motion_holder_id
        output_is_motion = source_input_ids == self.motion_holder_id_out
        type_ids = torch.zeros_like(source_input_ids)
        input_type_ind = source_input_ids >= self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size
        inputs_embeds = self.language_model.get_embeddings_from_ids(source_input_ids, type_ids)
        if motion_tokens is not None:
            motion_num_in_input = input_is_motion.sum(dim=-1) // self.holder_num_in_input
            for i, source_input_id in enumerate(source_input_ids):
                motion_embeddings = self.motion_und_head(motion_tokens[i])
                motion_embs = motion_embeddings[:motion_num_in_input[i]].split(self.llm_decoder_embed_dim, dim=-1)
                inputs_embeds[self.mod_id][i, input_is_motion[i], :] = torch.cat(motion_embs, dim=0)
        else:
            assert (~input_is_motion).all(), 'motion holder in input text, wihout provided motion tokens'
        if self.lm_type == 'encdec':
            target_inputs = self.tokenizer([self.output_motion_holder_seq], padding='max_length', max_length=self.motion_holder_repeat + 3, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
            labels_input_ids = target_inputs.input_ids.to(inputs_embeds.device)
            lables_attention_mask = target_inputs.attention_mask.to(inputs_embeds.device)
            output_is_motion = labels_input_ids == self.motion_holder_id_out
            output_type_ids = torch.zeros_like(labels_input_ids).to(inputs_embeds.device)
            output_type_ind = labels_input_ids >= self.text_vocab_size
            output_type_ids[output_type_ind] = 1
            ignore_ind = labels_input_ids == 0
            labels_input_ids[output_type_ind] -= self.text_vocab_size
            decoder_input_ids = self.language_model._shift_right(labels_input_ids)
            labels_input_ids[ignore_ind] = -100
            outputs = self.language_model(type_ids=type_ids, decoder_type_ids=output_type_ids, inputs_embeds=inputs_embeds, attention_mask=source_attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=lables_attention_mask.repeat(bsz, 1), output_hidden_states=True)
            hidden = outputs.decoder_hidden_states[-1][self.mod_id]
            latents = torch.stack([h[output_is_motion[i], :] for i, h in enumerate(hidden)])
        elif self.lm_type == 'dec':
            labels = source_input_ids.clone().detach()
            outputs = self.language_model(type_ids=type_ids, inputs_embeds=inputs_embeds, attention_mask=source_attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][self.mod_id]
            latents = torch.stack([h[output_is_motion[i], :] for i, h in enumerate(hidden)])
        if self.with_hid_norm:
            latents = self.norm_layer(latents)
            latents = latents + self.diffusion_pos_embed_learned
        return latents

    def generate_conditional(self, texts: Optional[List[str]]=None, motion_feats: Optional[Tensor]=None, motion_encode_net=None, lengths: Optional[List[int]]=None, task: str='t2m', with_len: bool=False, stage: str='train', tasks: dict=None):
        self.device = self.language_model.device
        if task in ['t2m', 'm2m', 'pred', 'inbetween']:
            if task == 't2m':
                assert texts is not None
                if not with_len:
                    if tasks is None:
                        tasks = [{'class': 't2m', 'input': ['Generate motion: <Caption_Placeholder>'], 'output': ['']}] * len(texts)
                    lengths = [0] * len(texts)
                else:
                    assert False, 'not Implemented'
                    tasks = [{'input': ['Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'], 'output': ['']}] * len(texts)
                motion_tokens = None
            elif task == 'pred':
                assert motion_feats is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{'class': 'pred', 'input': ['Predict motion: <Motion_Placeholder_s1>'], 'output': ['']}] * len(lengths)
                with torch.no_grad():
                    motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='pred')
            elif task == 'inbetween':
                assert motion_feats is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{'class': 'inbetween', 'input': ['Complete the masked motion: <Motion_Placeholder_Masked>'], 'output': ['']}] * len(lengths)
                with torch.no_grad():
                    motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='inbetween')
            inputs, outputs, modes = self.template_fulfill(tasks, lengths, texts, stage)
            outputs_tokens = self.generate_direct_motion([inn + ' \n' for inn in inputs], motion_tokens=motion_tokens)
            return outputs_tokens
        elif task in ['m2t', 't2t']:
            if task == 'm2t':
                assert motion_feats is not None and lengths is not None
                with torch.no_grad():
                    motion_tokens = self.motion_feats_to_tokens(motion_encode_net, motion_feats, lengths, modes='motion')
                if not with_len:
                    tasks = [{'class': 'm2t', 'input': ['Generate text: <Motion_Placeholder>'], 'output': ['']}] * len(lengths)
                else:
                    tasks = [{'class': 'm2t', 'input': ['Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'], 'output': ['']}] * len(lengths)
            elif task == 't2t':
                tasks = [{'class': 't2t', 'input': ['<Caption_Placeholder>'], 'output': ['']}] * len(lengths)
            texts = [''] * len(lengths)
            inputs, outputs, modes = self.template_fulfill(tasks, lengths, texts)
            outputs_tokens, cleaned_text = self.generate_direct([inn + ' \n' for inn in inputs], motion_tokens=motion_tokens, max_length=40, num_beams=1, do_sample=False, bad_words_ids=[[self.som_id], [self.eom_id]], gen_mode='text')
            return cleaned_text

    def sample_tokens(self, outputs, device, temperature=1.0, cfg=1.0, vae_mean_std_inv=None):
        if isinstance(outputs, List):
            output_to_sample = []
            motion_mask = torch.zeros(len(outputs))
            fake_embedding = torch.zeros((1, self.llm_decoder_embed_dim))
            for i, output in enumerate(outputs):
                if output is not None:
                    output_to_sample.append(output[:, :])
                else:
                    output_to_sample.append(fake_embedding)
                    motion_mask[i] = 1
            output_to_sample = torch.stack(output_to_sample, dim=0).to(device)
        else:
            motion_mask = torch.zeros(len(outputs))
            output_to_sample = outputs.to(device)
        if cfg > 1.0:
            output_to_sample = torch.cat([output_to_sample, self.fake_latent[None, :, :].repeat(output_to_sample.shape[0], 1, 1)], dim=0)
        sampled_token_latents = self.diffusion.sample(output_to_sample, temperature, cfg)
        if cfg > 1.0:
            sampled_token_latents, _ = sampled_token_latents.chunk(2, dim=0)
        if self.with_vae_latent_norm:
            sampled_token_latents = sampled_token_latents / vae_mean_std_inv
        return (sampled_token_latents, motion_mask)

def diffusion_cfg(self, predict, x, t, c, cfg_scale):
    half = x[:len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = predict(combined, t, c)
    eps, rest = (model_out[:, :self.in_channels], model_out[:, self.in_channels:])
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

class MotionGPT3DiffusionSampler:
    def __init__(self,bundle):
        self.bundle=bundle
        head=bundle.unwrap_network(bundle.diffusion_head)
        self.in_size=head.in_size;self.in_channels=head.in_channels
        self.gen_diffusion=create_diffusion(timestep_respacing=bundle.diffusion_config['num_sampling_steps'],**bundle.diffusion_options)
    def predict(self,*args,**kwargs):return self.bundle.predict_diffusion(*args,**kwargs)

    def sample(self, z, temperature=1.0, cfg=1.0):
        if not cfg == 1.0:
            bsz = z.shape[0] // 2
            if self.in_size is not None:
                noise = torch.randn(bsz, self.in_size, self.in_channels).cuda()
            else:
                noise = torch.randn(bsz, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = partial(diffusion_cfg, self, self.predict)
        else:
            if self.in_size is not None:
                noise = torch.randn(z.shape[0], self.in_size, self.in_channels).cuda()
            else:
                noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.predict
        sampled_token_latent = self.gen_diffusion.p_sample_loop(sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False, temperature=temperature)
        return sampled_token_latent
