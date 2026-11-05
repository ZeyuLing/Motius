"""Native readers for the original MotionStreamer TAE and token-training protocols."""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from motius.registry import DATASETS


@DATASETS.register_module()
class MotionStreamerTextTokenDataset(data.Dataset):

    def __init__(
        self,
        dataset_name,
        unit_length=4,
        latent_dir=None,
        hml_root="data/humanml3d_272",
        babel_root="data/babel_272",
        babel_stream_root="data/babel_272_stream",
    ):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        if dataset_name == "t2m_272":
            self.data_root = str(hml_root)
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            fps = 30
            self.max_motion_length = 78
            dim_pose = 272
            split_file = pjoin(self.data_root, "split", "train.txt")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(latent_dir, "%s.npy" % name))
            except:
                continue
            with cs.open(pjoin(self.text_dir, name + ".txt")) as f:
                text_data = []
                flag = False
                lines = f.readlines()
                for line in lines:
                    text_dict = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    t_tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    text_dict["caption"] = caption
                    text_dict["tokens"] = t_tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        if int(f_tag * fps / unit_length) < int(
                            to_tag * fps / unit_length
                        ):
                            m_token_list_new = [
                                m_token_list[
                                    int(f_tag * fps / unit_length) : int(
                                        to_tag * fps / unit_length
                                    )
                                ]
                            ]
                        if len(m_token_list_new) == 0:
                            continue
                        new_name = "%s_%f_%f" % (name, f_tag, to_tag)
                        data_dict[new_name] = {
                            "m_token_list": m_token_list_new,
                            "text": [text_dict],
                        }
                        new_name_list.append(new_name)
            if flag:
                data_dict[name] = {"m_token_list": m_token_list, "text": text_data}
                new_name_list.append(name)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = (data["m_token_list"], data["text"])
        m_tokens = np.array(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        if len(m_tokens.shape) == 3:
            m_tokens = m_tokens.squeeze(0)
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]
        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate(
                [
                    m_tokens,
                    np.zeros(
                        (self.max_motion_length - m_tokens_len, m_tokens.shape[1]),
                        dtype=int,
                    ),
                ],
                axis=0,
            )
        return dict(
            caption=caption, tokens=m_tokens, lengths=m_tokens_len, prefix_lengths=0
        )


@DATASETS.register_module()
class MotionStreamerStreamingTokenDataset(data.Dataset):

    def __init__(
        self,
        dataset_name,
        unit_length=4,
        latent_dir=None,
        hml_root="data/humanml3d_272",
        babel_root="data/babel_272",
        babel_stream_root="data/babel_272_stream",
    ):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        if dataset_name == "t2m_babel_272":
            self.babel_stream_data_root = str(babel_stream_root)
            self.babel_stream_text_dir = pjoin(
                self.babel_stream_data_root, "train_stream_text"
            )
            fps = 30
            self.max_motion_length = 78
            self.hml_data_root = str(hml_root)
            self.hml_text_dir = pjoin(self.hml_data_root, "texts")
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        id_list = []
        for file in os.listdir(latent_dir):
            if file.endswith(".npy"):
                id_list.append(file[:-4])
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            m_token_list = np.load(pjoin(latent_dir, "%s.npy" % name))
            if len(m_token_list) > self.max_motion_length:
                continue
            if name.split("_")[0] == "seq":
                with cs.open(pjoin(self.babel_stream_text_dir, name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        text_dict = {}
                        B_split = line.strip().split("*")[1].split("#")
                        B_text = line.strip().split("*")[1].split("#")[0]
                        if B_text == "":
                            continue
                        B_t_tokens = B_split[1].split(" ")
                        A_motion_length = B_split[-1]
                        A_token_length = int(A_motion_length) // unit_length
                        text_dict["caption"] = B_text
                        text_dict["tokens"] = B_t_tokens
                        flag = True
                        text_data.append(text_dict)
            else:
                with cs.open(pjoin(self.hml_text_dir, name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        t_tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        A_token_length = 0
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = t_tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            if int(f_tag * fps / unit_length) < int(
                                to_tag * fps / unit_length
                            ):
                                m_token_list_new = [
                                    m_token_list[
                                        int(f_tag * fps / unit_length) : int(
                                            to_tag * fps / unit_length
                                        )
                                    ]
                                ]
                            if len(m_token_list_new) == 0:
                                continue
                            new_name = "%s_%f_%f" % (name, f_tag, to_tag)
                            data_dict[new_name] = {
                                "m_token_list": m_token_list_new,
                                "text": [text_dict],
                                "A_token_length": A_token_length,
                            }
                            new_name_list.append(new_name)
            if flag:
                data_dict[name] = {
                    "m_token_list": m_token_list,
                    "text": text_data,
                    "A_token_length": A_token_length,
                }
                new_name_list.append(name)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = (data["m_token_list"], data["text"])
        m_tokens = np.array(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        if len(m_tokens.shape) == 3:
            m_tokens = m_tokens.squeeze(0)
        A_token_length = data["A_token_length"]
        m_tokens_len = m_tokens.shape[0]
        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate(
                [
                    m_tokens,
                    np.zeros(
                        (self.max_motion_length - m_tokens_len, m_tokens.shape[1]),
                        dtype=int,
                    ),
                ],
                axis=0,
            )
        return dict(
            caption=caption,
            tokens=m_tokens,
            lengths=m_tokens_len,
            prefix_lengths=A_token_length,
        )


@DATASETS.register_module()
class MotionStreamerTAEDataset(data.Dataset):

    def __init__(
        self,
        dataset_name,
        window_size=64,
        unit_length=4,
        hml_root="data/humanml3d_272",
        babel_root="data/babel_272",
        babel_stream_root="data/babel_272_stream",
    ):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        if dataset_name == "t2m_272":
            self.data_root = str(hml_root)
            self.motion_dir = pjoin(self.data_root, "motion_data")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 300
            self.meta_dir = pjoin(self.data_root, "mean_std")
            split_file = pjoin(self.data_root, "split", "train.txt")
        elif dataset_name == "t2m_babel_272":
            self.hml_data_root = str(hml_root)
            self.hml_motion_dir = pjoin(self.hml_data_root, "motion_data")
            hml_split_file = pjoin(self.hml_data_root, "split", "train.txt")
            self.joints_num = 22
            self.max_motion_length = 300
            self.babel_data_root = str(babel_root)
            self.babel_motion_dir = pjoin(self.babel_data_root, "motion_data")
            babel_split_file = pjoin(self.babel_data_root, "split", "train.txt")
            self.meta_dir = pjoin(self.babel_data_root, "t2m_babel_mean_std")
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
        mean = np.load(pjoin(self.meta_dir, "Mean.npy"))
        std = np.load(pjoin(self.meta_dir, "Std.npy"))
        self.data = []
        self.lengths = []
        id_list = []
        if dataset_name == "t2m_272":
            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    id_list.append(line.strip())
        elif dataset_name == "t2m_babel_272":
            with cs.open(hml_split_file, "r") as f:
                for line in f.readlines():
                    id_list.append(line.strip())
            with cs.open(babel_split_file, "r") as f:
                for line in f.readlines():
                    id_list.append("b_" + line.strip())
        for name in tqdm(id_list):
            try:
                if dataset_name == "t2m_272":
                    motion = np.load(pjoin(self.motion_dir, name + ".npy"))
                elif dataset_name == "t2m_babel_272":
                    if name.split("_")[0] == "b":
                        motion = np.load(
                            pjoin(self.babel_motion_dir, name.split("_")[1] + ".npy")
                        )
                    else:
                        motion = np.load(pjoin(self.hml_motion_dir, name + ".npy"))
                else:
                    raise ValueError(f"Dataset {dataset_name} not found")
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                pass
        print(f"Training on {len(self.data)} motion sequences...")
        self.mean = mean
        self.std = std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx : idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return dict(motion=motion)
