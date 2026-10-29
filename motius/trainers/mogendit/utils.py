import torch
import numpy as np
import random
import time

def random_index(data_len:int, sampling_rate=1.0, seed:int=None) -> list:
    """
    随机采样索引的函数

    参数:
    sample_size (int): 样本数量
    sampling_rate (float): 采样率，范围在(0, 1]
    seed (int or None): 随机数生成的种子，如果为None则不设置种子

    返回:
    list: 随机采样的索引列表
    """
    if not (0 < sampling_rate <= 1):
        raise ValueError("采样率必须在(0, 1]范围内")

    # 设置随机数种子
    if seed is not None:
        rng = np.random.RandomState(seed)


    else:
        rng = np.random

    # 计算需要采样的样本数量
    num_samples_to_select = max(int(data_len * sampling_rate), 1)

    # 生成样本索引的随机排列
    all_indices = np.arange(data_len)
    rng.shuffle(all_indices)

    # 从随机排列的索引中选择需要的数量
    selected_indices = all_indices[:num_samples_to_select]

    return selected_indices