# -*- coding: utf-8 -*-
"""
utils.py
存放通用功能函数 (可选)
"""

import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    固定随机种子以保证可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
