import re
from abc import ABC
from typing import List, Type, Union, Tuple

import torch
from torch import Tensor
from mmengine import print_log

# default keys for each modality. The Dataset module will search for these keys in the annotation files.
motion_keys = ["motion"]
caption_keys = ["caption"]
audio_keys = ["audio"]
music_keys = ["music"]
duration_keys = ["duration"]
script_keys = ["speech_script"]
genre_keys = ["genre"]


"""
    Some special tokens representing different modal prompt sequence
"""


class Modality(ABC):
    """
    Abstract class to define the utilized modalities of Multi-modal Motion tasks.
    name: the name of the modality
    token_format: used in Motion LLM, defines the string format of multi-modal tokens. Only motion and audio used it.
    holder: placeholder of the modality showing in Prompt templates
    bos: special token which stands for begin of modality substring in conversation
    eos: special token which stands for end of modality substring in conversation
    data_keys: Keys to save the modality into DataSample
    load_keys: Keys to load the modality from annotation files
    """

    name = None
    token_format = "<|MODAL_{}|>"
    bos = None
    eos = None
    data_keys = None  # the keys load the modality from data batch dict
    load_keys = None  # the keys to load the modality from annotation files

    @classmethod
    def locatable(cls):
        if (
            cls.bos is not None
            and cls.eos is not None
            and len(cls.bos)
            and len(cls.eos)
        ):
            return True
        return False

    @classmethod
    def index_to_string(cls, idx: List[int]):
        """For motion and audio
        :param idx: index list.
        :return: for exp, <|MOTION_13|><|MOTION_150|><|MOTION_2|>
        """
        token_string = [cls.token_format.format(int(i)) for i in idx]
        return "".join(token_string)

    @classmethod
    def string_to_index(
        cls, string: str, return_tensor=True
    ) -> Union[List[int], Tensor, None]:
        if string is None:
            return None
        if cls.token_format is None:
            raise ValueError(
                f"Modality {type(cls)} doesnt support string to index, u can encode it with tokenizer"
            )
        pattern = re.escape(cls.token_format).replace("\\{\\}", r"(\d+)")
        ids = re.findall(pattern, string)
        ids = [int(i) for i in ids]
        if return_tensor:
            ids = torch.tensor(ids, dtype=torch.int64)
        return ids

    @classmethod
    def locate_modality(cls, text: str) -> List[str]:
        """Return all substrings between cls.bos and cls.eos in text."""
        pattern = re.compile(
            re.escape(cls.bos) + r"(.*?)" + re.escape(cls.eos),
            flags=re.DOTALL,  # 支持跨多行
        )
        substrings = pattern.findall(text)

        # 过滤掉只包含省略号/标点/空白的匹配
        # 要求至少包含一个字母 / 数字 / 中文
        substrings = [
            s for s in substrings if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s)
        ]

        return substrings


class Motion(Modality):
    name = "motion"
    token_format = "<|MOTION_{}|>"
    bos = "<|begin_of_motion|>"
    eos = "<|end_of_motion|>"
    data_keys = motion_keys
    load_keys = motion_keys
    mp_separator = "<|next_person|>"

    @classmethod
    def string_to_index(
        cls, string: str, return_tensor=True
    ) -> Union[List[int], Tensor, None]:
        """Convert motion string to motion codes

        :param string: motion string
        :param return_tensor: whether to return a tensor, defaults to True
        :return: motion codes. in shape [n] or [p n]
        """

        def _string_to_index(string: str, return_tensor: bool):
            # super
            if string is None:
                return None
            pattern = re.escape(cls.token_format).replace("\\{\\}", r"(\d+)")
            ids = re.findall(pattern, string)
            ids = [int(i) for i in ids]
            if return_tensor:
                ids = torch.tensor(ids, dtype=torch.int64)
            return ids

        # first separate the string into multiple motion strings
        if string is None:
            return None
        motion_strings = string.split(cls.mp_separator)
        motion_codes = [
            _string_to_index(motion_string, return_tensor)
            for motion_string in motion_strings
        ]
        # for multi-person scene, only the minimum length motion code is used
        min_length = min([len(code) for code in motion_codes])
        # if length of each person's motion, use the minimum length and make warning
        if not all([len(code) == min_length for code in motion_codes]):
            print_log(
                f"Not all person's motion token length is {min_length}, "
                f"get {[len(code) for code in motion_codes]}. "
                f"Use the minimum length {min_length} instead."
            )

        motion_codes = [code[:min_length] for code in motion_codes]
        if return_tensor:
            motion_codes = torch.stack(motion_codes, dim=0)
        return motion_codes

    @classmethod
    def index_to_string(cls, idx: torch.Tensor):
        """For motion
        :param idx: index list. [n] or [p n]
        :return: for exp, <|MOTION_13|><|MOTION_150|><|MOTION_2|>
        """
        if idx.ndim > 1 and idx.shape[0] > 1:
            # multi-person motion. Unless the
            token_string = []
            for person_idx, person in enumerate(idx):
                person_string = cls.mp_separator if person_idx != 0 else ""
                person_string = person_string + "".join(
                    [cls.token_format.format(int(i)) for i in person]
                )
                token_string.append(person_string)
        else:
            token_string = [cls.token_format.format(int(i)) for i in idx]
        return "".join(token_string)


class PastMotion(Motion):
    name = "past_motion"
    bos = "<|begin_of_past_motion|>"
    eos = "<|end_of_past_motion|>"
    data_keys = [f"past_{key}" for key in motion_keys]


class MiddleMotion(Motion):
    name = "middle_motion"
    bos = "<|begin_of_middle_motion|>"
    eos = "<|end_of_middle_motion|>"
    data_keys = [f"middle_{key}" for key in motion_keys]


class FutureMotion(Motion):
    name = "future_motion"
    bos = "<|begin_of_future_motion|>"
    eos = "<|end_of_motion|>"
    data_keys = [f"future_{key}" for key in motion_keys]


class Audio(Modality):
    name = "audio"
    token_format = "<|AUDIO_{}|>"
    bos = "<|begin_of_audio|>"
    eos = "<|end_of_audio|>"
    data_keys = audio_keys
    load_keys = audio_keys


class Music(Audio):
    name = "music"
    data_keys = music_keys
    load_keys = music_keys
    bos = "<|begin_of_music|>"
    eos = "<|end_of_music|>"


class PastMusic(Music):
    name = "past_music"
    bos = "<|begin_of_past_music|>"
    eos = "<|end_of_past_music|>"
    data_keys = [f"past_{key}" for key in music_keys]


class FutureMusic(Music):
    name = "future_music"
    bos = "<|begin_of_future_music|>"
    eos = "<|end_of_music|>"
    data_keys = [f"future_{key}" for key in music_keys]


class Text(Modality):
    name = "text"
    token_format = None
    bos = ""
    eos = ""


class Caption(Text):
    name = "caption"
    bos = "<|begin_of_caption|>"
    eos = "<|end_of_caption|>"
    data_keys = caption_keys
    load_keys = caption_keys


class SpeechScript(Text):
    name = "speech_script"
    bos = "<|begin_of_script|>"
    eos = "<|end_of_script|>"
    data_keys = script_keys
    load_keys = script_keys


class Duration(Text):
    name = "duration"
    bos = "<|begin_of_duration|>"
    eos = "<|end_of_duration|>"
    data_keys = duration_keys
    load_keys = duration_keys


class Genre(Text):
    name = "genre"
    bos = "<|begin_of_genre|>"
    eos = "<|end_of_genre|>"
    data_keys = genre_keys
    load_keys = genre_keys


class NumPerson(Text):
    name = "num_person"
    bos = "<|begin_of_num_person|>"
    eos = "<|end_of_num_person|>"
    data_keys = ["num_person"]
    load_keys = ["num_person"]


def is_modal(
    modal_A: Type[Modality], modal_B: Union[Type[Modality], Tuple[Type[Modality], ...]]
) -> bool:
    return issubclass(modal_A, modal_B)
