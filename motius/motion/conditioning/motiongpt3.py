"""MotionGPT3 task formatting and continuous-latent conditioning."""
import torch,random,math
import numpy as np
from typing import List,Union,Optional
from torch import Tensor

class MotionGPT3Processor:
    def __init__(self,tokenizer,options,device):
        self.tokenizer=tokenizer;self.device=device
        for key,value in options.items():setattr(self,key,value)
        self.lm_type='dec'
        self.motion_holder_output_word='<motion_latent_holder_output>';self.motion_holder_word='<motion_latent_holder>'
        self.som_word,self.eom_word='<start_of_motion>','<end_of_motion>'
        self.mom_word,self.pom_word='<masked_motion>','<pad_motion>'
        self.tokenizer.add_special_tokens({'additional_special_tokens':[self.som_word,self.eom_word]})
        self.text_vocab_size=len(self.tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens':[self.mom_word,self.pom_word]})
        self.tokenizer.add_tokens([self.motion_holder_output_word,self.motion_holder_word])
        self.som_id,self.eom_id=self.tokenizer.convert_tokens_to_ids([self.som_word,self.eom_word])
        self.motion_holder_id_out,self.motion_holder_id=self.tokenizer.convert_tokens_to_ids([self.motion_holder_output_word,self.motion_holder_word])
        self.tokenizer.pad_token=self.tokenizer.eos_token;self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        a=self.motion_holder_output_word*self.motion_holder_repeat;b=self.motion_holder_word*self.holder_num_in_input
        if self.motion_holder_seq_mode=='withse':a=self.som_word+a+self.eom_word;b=self.som_word+b+self.eom_word
        self.output_motion_holder_seq=a;self.input_motion_holder_seq=b;self.masked_holder_seq=self.mom_word*self.motion_holder_repeat

    def motion_feats_to_tokens(self, motion_encode_net, motion_feats: Tensor, lengths: List[int], modes='motion', motion_tokens=None):
        motion_tokens = []
        bs, _, feat_dim = motion_feats.shape
        if isinstance(modes, str):
            modes = [modes] * bs
        for i in range(bs):
            mode = modes[i]
            length = lengths[i]
            max_len = length
            motion_feat = motion_feats[i, :length, :]
            motion_feats_to_encode = torch.zeros((3, length, feat_dim), requires_grad=False)
            if mode in ['motion', 't2m', 'm2t', 'l2m', 'n2m', 'm2l']:
                motion_feats_to_encode[0, :, :] = motion_feat
                lengths_input = [length]
            elif mode in ['predict', 'pred']:
                predict_head = int(length * self.predict_ratio)
                motion_feats_to_encode[0, :predict_head, :] = motion_feat[:predict_head, :]
                motion_feats_to_encode[1, :length - predict_head, :] = motion_feat[predict_head:, :]
                lengths_input = [predict_head, length - predict_head]
                max_len = max(predict_head, length - predict_head)
            elif mode in ['inbetween', 'inbetweening']:
                masked_head = int(length * self.inbetween_ratio)
                masked_tail = int(length * (1 - self.inbetween_ratio))
                motion_feats_to_encode[0, :masked_head, :] = motion_feat[:masked_head, :]
                motion_feats_to_encode[1, :length - masked_tail, :] = motion_feat[masked_tail:, :]
                motion_feats_to_encode[2, :, :] = motion_feat
                lengths_input = [masked_head, length - masked_tail, length]
            elif mode in ['t2t', 'n2t']:
                motion_tokens.append(None)
                continue
            else:
                assert False, mode
            motion_feats_to_encode = motion_feats_to_encode[:len(lengths_input), :max_len]
            dist = motion_encode_net.encode_dist(motion_feats_to_encode.to(motion_feats.device), lengths_input)
            z, _ = motion_encode_net.encode_dist2z(dist)
            if self.with_vae_latent_norm:
                zz = z.permute(1, 0, 2).mul_(motion_encode_net.mean_std_inv)
            else:
                zz = z.permute(1, 0, 2)
            motion_tokens.append(zz)
        return motion_tokens

    def clean_text_string(self, motion_string: List[str]):
        output_string = []
        for i in range(len(motion_string)):
            output_string.append(motion_string[i].replace(self.output_motion_holder_seq, '<Motion_Placeholder>').replace(self.input_motion_holder_seq, '<Motion_Placeholder>').replace(self.tokenizer.pad_token, '').replace('\n', ' '))
        return (None, output_string)

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str):
        seconds = math.floor(length / self.framerate)
        motion_masked = motion_string + self.masked_holder_seq + motion_string
        if random.random() < self.quota_ratio and '"' not in text:
            text = f'"{text}"'
        prompt = prompt.replace('<Caption_Placeholder>', text).replace('<Motion_Placeholder>', motion_string).replace('<Frame_Placeholder>', f'{length}').replace('<Second_Placeholder>', '%.1f' % seconds).replace('<Motion_Placeholder_s1>', motion_string).replace('<Motion_Placeholder_s2>', motion_string).replace('<Motion_Placeholder_Masked>', motion_masked)
        if 'Placeholder' in prompt:
            startIndex = prompt.index('Placeholder')
            assert False, 'found unsupported PlaceHolder' + prompt[startIndex - 8:startIndex + 20]
        return prompt

    def template_fulfill(self, tasks, lengths, texts, stage='test'):
        inputs = []
        outputs = []
        modes = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(self.placeholder_fulfill(input_template, length, self.input_motion_holder_seq, texts[i]))
            outputs.append(self.placeholder_fulfill(output_template, length, self.output_motion_holder_seq, texts[i]))
            modes.append(tasks[i]['class'])
        return (inputs, outputs, modes)

    def random_spans_noise_mask(self, length):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
                Args:
                    num_items: an integer scalar > 0
                    num_segments: an integer scalar in [1, num_items]
                Returns:
                    a Tensor with shape [num_segments] containing positive integers that add
                    up to num_items
                """
            mask_indices = np.arange(num_items - 1) < num_segments - 1
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length
        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = np.reshape(np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2])
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids.to('cpu'))
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate([input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1)
        input_ids = torch.tensor(input_ids, device=self.device)
        return input_ids
