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

        
        if model_type == "t5":
            # lm = T5ForConditionalGeneration.from_pretrained(model_path)
            from transformers.models.t5.modeling_t5 import T5Config
            from transformers.generation.configuration_utils import GenerationConfig
            from motius.models.motiongpt3.network.mot_code.mot_example_t5 import MoTT5ForConditionalGeneration
            mconfig = T5Config.from_pretrained(f'{model_path}/config.json')
            # import json
            # with open('deps/mot-t5-base/generation_config.json', 'r') as file:
            #     gconfig = GenerationConfig(json.load(file))
            language_model = MoTT5ForConditionalGeneration(mconfig, motion_codebook_size=motion_codebook_size+4)
            # state_dict = torch.load(f'{model_path}/model_state_dict.pth')
            # # state_dict = torch.load('deps/mot-t5-base/mot_model_state_dict.pth')
           
            # Instantiate tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
            # tokenizer.pad_token_id = 0
            tokenizer.bos_token_id = 0
            
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            # self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            from transformers.models.gpt2.modeling_gpt2 import GPT2Config
            from motius.models.motiongpt3.network.mot_code.mot_example_gpt2 import MoTGPT2LMHeadModel
            mconfig = GPT2Config.from_pretrained(f'{model_path}/config.json')
            # import json
            # with open('deps/mot-t5-base/generation_config.json', 'r') as file:
            #     gconfig = GenerationConfig(json.load(file))
            language_model = MoTGPT2LMHeadModel(mconfig, motion_codebook_size=motion_codebook_size+4)
            # # state_dict = torch.load('deps/mot-t5-base/model_state_dict.pth')
            # state_dict = torch.load(f'{model_path}/model_state_dict.pth')
            
            # Instantiate tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")


        # Instantiate tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        # self.tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.text_vocab_size = len(self.tokenizer)
        # Add motion tokens
        self.tokenizer.add_tokens(
            [f'<motion_id_{i}>' for i in range(self.m_codebook_size)])
        # self.tokenizer.add_tokens(['<start_of_motion>', '<end_of_motion>', '<masked_motion>'])
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<start_of_motion>', '<end_of_motion>', '<masked_motion>', '<pad_motion>']})
        
        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            # self.tokenizer.padding_side = "left"

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
        
        self.language_model.config.mot_lm_dim = self.language_model.config.motion_vocab_size
        print('mot_lm_dim', self.language_model.config.mot_lm_dim)
        self.language_model.config.text_vocab_size = self.text_vocab_size
        self.language_model.set_modality_info(self.tokenizer)
        # state_dict_1 = torch.load(f'{model_path}/motiongpt_pretrained.ckpt')
        # msg = language_model.load_state_dict(state_dict_1, strict=False)
        # if len(msg.unexpected_keys)>0:
        #     print('1- unexpected_keys keys:', msg.unexpected_keys)
        #     exit()
        # print('1- missing keys:', msg.missing_keys)
        self.mod2id = self.language_model.mod2id.copy()
        # self.language_model.modality_infos[1].post_processor
        # self.language_model.init_mod_token_embeddings(mod_id=1)
        # # new_num_tokens = (len(self.tokenizer))
        # old_embeddings = self.language_model.pre_processors[1]
        # old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        # new_embeddings = self.language_model.pre_processors[1]
        # added_num_tokens = new_embeddings.weight.shape[0]
        # # print('added_num_tokens', added_num_tokens)
        # self.language_model._init_added_embeddings_weights_with_mean(
        #             old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
        #         )

        # old_lm_head = self.language_model.get_output_embeddings()
        # old_num_tokens, old_lm_head_dim = old_lm_head.weight.size() 
        # new_lm_head = self.language_model.post_processors.fn[1]
        # added_num_tokens = new_lm_head.weight.shape[0]
        # # print('added_num_tokens', added_num_tokens)
        # self.language_model._init_added_lm_head_weights_with_mean(
        #             old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed=False
        #         )
        
        
        for n, p in self.language_model.named_parameters():
            if n in state_dict.keys() or 'fn.0' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
                # print(n, end=' ')
        self.language_model.pre_processors[1].weight.requires_grad = True
        self.language_model.post_processors[1].weight.requires_grad = True

        # if new_token_type == "insert":
        #     self.language_model.resize_token_embeddings(len(self.tokenizer))
        # elif new_token_type == "mlp":
        #     shared = NewTokenEmb(self.language_model.shared,
        #                          self.m_codebook_size + 3)
        #     # lm_head = NewTokenEmb(self.language_model.lm_head,
        #     #   self.m_codebook_size + 3)
        #     self.language_model.resize_token_embeddings(len(self.tokenizer))
        #     self.language_model.shared = shared
        #     # self.language_model.lm_head = lm_head

        # Lora
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
            from peft.utils.other import fsdp_auto_wrap_policy
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                #  inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05)
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)

    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)
        source_attention_mask = source_encoding.attention_mask.to(motion_tokens.device)

        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(
                mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(
                target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids,
                                                     target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids,
                                                     input_ids_sentinel)

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = target_inputs.attention_mask.to(
                motion_tokens.device)
            
        type_ids = torch.zeros_like(source_input_ids).to(motion_tokens.device)
        input_type_ind = source_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size

        output_type_ids = torch.zeros_like(labels_input_ids).to(motion_tokens.device)
        output_type_ind = labels_input_ids>=self.text_vocab_size
        output_type_ids[output_type_ind] = 1
        ignore_ind = labels_input_ids == 0
        # output_type_ids[labels_input_ids == -100] = -1
        labels_input_ids[output_type_ind] -= self.text_vocab_size
        decoder_input_ids=self.language_model._shift_right(labels_input_ids)

        labels_input_ids[ignore_ind] = -100
        # motion_info = self.language_model.modality_infos[1]
        # som_id, eom_id = motion_info.som_id, motion_info.eom_id
        # msk_id = motion_info.msk_id
        # output_type_ids[labels_input_ids==msk_id] = 1
        # for input_id in labels_input_ids:
        #     if som_id in input_id:
        #         s_ind = input_id.index(som_id)
        #         e_ind = input_id.index(eom_id)
        #         output_type_ids[s_ind:e_ind] = 1

        outputs = self.language_model(
            type_ids=type_ids,
            decoder_type_ids=output_type_ids,
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,
            labels=labels_input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=lables_attention_mask
            if condition == 'supervised' else None,
        )

        torch.cuda.empty_cache()
        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):
        # assert False
        self.tokenizer.padding_side = "right"
        # self.tokenizer.padding_side = "left"

        with torch.no_grad():
            # Tensor to string
            motion_strings = self.motion_token_to_string(motion_tokens, lengths)

            # Supervised or unsupervised
            condition = random.choice(
                ['motion', 'supervised', 'supervised', 'supervised'])

            if condition == 'text':
                labels = texts
            elif condition == 'motion':
                labels = motion_strings
            else:
                inputs, outputs = self.template_fulfill(tasks, lengths,
                                                        motion_strings, texts)
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

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        # type_ids = torch.ones_like(labels_input_ids).to(motion_tokens.device)
        type_ids = torch.zeros_like(labels_input_ids).to(motion_tokens.device)
        input_type_ind = labels_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        labels_input_ids[input_type_ind] -= self.text_vocab_size

        outputs = self.language_model(
            type_ids=type_ids,
            input_ids=labels_input_ids,
            attention_mask=lables_attention_mask,
            labels=labels_input_ids,
            )

        torch.cuda.empty_cache()
        return outputs

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None,
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
        type_ids = torch.zeros_like(source_input_ids)
        input_type_ind = source_input_ids>=self.text_vocab_size
        type_ids[input_type_ind] = 1
        source_input_ids[input_type_ind] -= self.text_vocab_size
        
        mid = self.mod2id[gen_mode]
        som_id = self.language_model.modality_infos[mid].som_id
        eom_id = self.language_model.modality_infos[mid].eom_id + self.language_model.modality_infos[mid].token_id_start
        # decoder_input_ids = torch.full((source_input_ids.shape[0], 1), som_id).to(source_input_ids)
        # decoder_type_ids = torch.full((source_input_ids.shape[0], 1), mid).to(source_input_ids)

        with torch.no_grad():
            if self.lm_type == 'encdec':
                outputs = self.language_model.generate(
                    type_ids=type_ids,
                    input_ids=source_input_ids,
                    attention_mask=source_attention_mask,
                    # decoder_input_ids=decoder_input_ids,
                    # decoder_type_ids=decoder_type_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    bad_words_ids=bad_words_ids,
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

                source_attention_mask = source_encoding.attention_mask.to(self.device)
                outputs = self.language_model.generate(
                    type_ids=type_ids,
                    input_ids=source_input_ids,
                    attention_mask=source_attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    max_new_tokens=max_length,
                    mode=gen_mode,
                    # repetition_penalty=1.1,
                    )
                # self.tokenizer.padding_side = 'left'

        # outputs1 = torch.cat([outputs, torch.full((*outputs.shape[:-1],1), eom_id, device=self.device, dtype=outputs.dtype)], dim=-1)
        outputs_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        outputs_tokens, cleaned_text = self.motion_string_to_token(
            outputs_string)

        return outputs_tokens, cleaned_text

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{
                            'input':
                            ['Generate motion: <Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)

                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
                    
            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(
                    motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append(
                        '>'.join(motion_strings_old[i].split('>')[:split]) +
                        '>')

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(
                    motion_tokens, lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts,
                                                    stage)

            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=96,
                num_beams=1,
                do_sample=True,
                gen_mode='motion'
                )

            return outputs_tokens

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths)

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                gen_mode='text',
                # bad_words_ids=self.bad_words_ids
            )
            return cleaned_text

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                ('<start_of_motion>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 '<end_of_motion>'))
                # (f'<motion_id_{self.m_codebook_size}>' +
                #  ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                #  f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                # motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                # f'<motion_id_{self.m_codebook_size + 1}>')
                motion_string[i], '<start_of_motion>', '<end_of_motion>')
            string_list = string.split('><')
            token_list = [
                int(i.split('_')[-1].replace('>', ''))
                for i in string_list[1:-1] if i.split('_')[-1] != 'motion'  # '<masked_motion>'
            ]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>').replace(
                self.tokenizer.pad_token, '').replace(
                '\n', ' '))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]) + '><end_of_motion>'
        motion_predict_last = '<start_of_motion>' + '>'.join(
            motion_splited[predict_head:])
        motion_masked = '>'.join(
            motion_splited[:masked_head]) + '>' + '<masked_motion>'* (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])
        
        # motion_predict_head = '>'.join(
        #     motion_splited[:predict_head]
        # ) + f'><motion_id_{self.m_codebook_size+1}>'
        # motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
        #     motion_splited[predict_head:])
        # motion_masked = '>'.join(
        #     motion_splited[:masked_head]
        # ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
        #     masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',
            motion_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i]))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return '<start_of_motion><motion_id_0><end_of_motion>'
            # return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return startStr + content[startIndex:endIndex] + endStr

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
