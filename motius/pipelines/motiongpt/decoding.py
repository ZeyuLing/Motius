"""MotionGPT autoregressive decoding and released task prompts."""
from typing import List,Optional
import torch
from torch import Tensor
from motius.motion.conditioning.motiongpt import MotionGPTTextProcessor


class MotionGPTDecoder(MotionGPTTextProcessor):
    def __init__(self,bundle):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(bundle.t5_model_path,legacy=True,local_files_only=bundle.local_files_only)
        tokenizer.add_tokens([f'<motion_id_{i}>' for i in range(bundle.codebook_size+3)])
        super().__init__(tokenizer,bundle.device,motion_codebook_size=bundle.codebook_size)
        self.bundle=bundle

    @property
    def language_model(self):
        return self.bundle.unwrap_network(self.bundle.lm).language_model

    def generate_direct(self, texts: List[str], max_length: int=256, num_beams: int=1, do_sample: bool=True, bad_words_ids: List[int]=None):
        self.device = self.language_model.device
        if self.lm_type == 'dec':
            texts = [text + ' \n ' for text in texts]
        source_encoding = self.tokenizer(texts, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(source_input_ids, max_length=max_length, num_beams=num_beams, do_sample=do_sample, bad_words_ids=bad_words_ids)
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(input_ids=source_input_ids, attention_mask=source_attention_mask, pad_token_id=self.tokenizer.pad_token_id, do_sample=do_sample, max_new_tokens=max_length)
            self.tokenizer.padding_side = 'left'
        outputs_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string)
        return (outputs_tokens, cleaned_text)

    def generate_conditional(self, texts: Optional[List[str]]=None, motion_tokens: Optional[Tensor]=None, lengths: Optional[List[int]]=None, task: str='t2m', with_len: bool=False, stage: str='train', tasks: dict=None):
        self.device = self.language_model.device
        if task in ['t2m', 'm2m', 'pred', 'inbetween']:
            if task == 't2m':
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{'input': ['Generate motion: <Caption_Placeholder>'], 'output': ['']}] * len(texts)
                    lengths = [0] * len(texts)
                else:
                    tasks = [{'input': ['Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'], 'output': ['']}] * len(texts)
            elif task == 'pred':
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{'input': ['Predict motion: <Motion_Placeholder_s1>'], 'output': ['']}] * len(lengths)
                motion_strings_old = self.motion_token_to_string(motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append('>'.join(motion_strings_old[i].split('>')[:split]) + '>')
            elif task == 'inbetween':
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{'input': ['Complete the masked motion: <Motion_Placeholder_Masked>'], 'output': ['']}] * len(lengths)
                motion_strings = self.motion_token_to_string(motion_tokens, lengths)
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts, stage)
            outputs_tokens, cleaned_text = self.generate_direct(inputs, max_length=128, num_beams=1, do_sample=True)
            return outputs_tokens
        elif task == 'm2t':
            assert motion_tokens is not None and lengths is not None
            motion_strings = self.motion_token_to_string(motion_tokens, lengths)
            if not with_len:
                tasks = [{'input': ['Generate text: <Motion_Placeholder>'], 'output': ['']}] * len(lengths)
            else:
                tasks = [{'input': ['Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'], 'output': ['']}] * len(lengths)
            texts = [''] * len(lengths)
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(inputs, max_length=40, num_beams=1, do_sample=False)
            return cleaned_text
