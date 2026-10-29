"""Released MotionGPT text/motion formatting and span corruption."""
from typing import List,Optional
import random,math
import numpy as np
import torch
from torch import Tensor


class MotionGPTTextProcessor:
    def __init__(self,tokenizer,device,*,motion_codebook_size=512,max_length=256,
                 framerate=20.0,down_t=4,predict_ratio=.2,inbetween_ratio=.25,
                 noise_density=.15,mean_noise_span_length=3,quota_ratio=.5):
        self.tokenizer=tokenizer;self.device=torch.device(device)
        self.m_codebook_size=motion_codebook_size;self.max_length=max_length
        self.framerate=framerate;self.down_t=down_t;self.predict_ratio=predict_ratio
        self.inbetween_ratio=inbetween_ratio;self.noise_density=noise_density
        self.mean_noise_span_length=mean_noise_span_length;self.quota_ratio=quota_ratio
        self.lm_type='encdec'

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(f'<motion_id_{self.m_codebook_size}>' + ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) + f'<motion_id_{self.m_codebook_size + 1}>')
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(f'<motion_id_{self.m_codebook_size}>' + ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) + f'<motion_id_{self.m_codebook_size + 1}>')
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(motion_string[i], f'<motion_id_{self.m_codebook_size}>', f'<motion_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            token_list = [int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1]]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list, dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))
        return (motion_tokens, output_string)

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str):
        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        motion_predict_head = '>'.join(motion_splited[:predict_head]) + f'><motion_id_{self.m_codebook_size + 1}>'
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(motion_splited[predict_head:])
        motion_masked = '>'.join(motion_splited[:masked_head]) + '>' + f'<motion_id_{self.m_codebook_size + 2}>' * (masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])
        if random.random() < self.quota_ratio:
            text = f'"{text}"'
        prompt = prompt.replace('<Caption_Placeholder>', text).replace('<Motion_Placeholder>', motion_string).replace('<Frame_Placeholder>', f'{length}').replace('<Second_Placeholder>', '%.1f' % seconds).replace('<Motion_Placeholder_s1>', motion_predict_head).replace('<Motion_Placeholder_s2>', motion_predict_last).replace('<Motion_Placeholder_Masked>', motion_masked)
        return prompt

    def template_fulfill(self, tasks, lengths, motion_strings, texts, stage='test'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(self.placeholder_fulfill(input_template, length, motion_strings[i], texts[i]))
            outputs.append(self.placeholder_fulfill(output_template, length, motion_strings[i], texts[i]))
        return (inputs, outputs)

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size + 1}>'
        return f'<motion_id_{self.m_codebook_size}>' + content[startIndex:endIndex] + f'<motion_id_{self.m_codebook_size + 1}>'

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
