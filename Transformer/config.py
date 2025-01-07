# config.py
# -*- coding: utf-8 -*-

import torch

class Config:
    data_dir = "./rt-polaritydata"
    pos_file = "rt-polarity.pos"
    neg_file = "rt-polarity.neg"

    # 数据划分
    train_ratio = 0.8
    val_ratio = 0.2


    # 词表配置
    min_freq = 2

    # 模型超参数（建议先小一点，以防止过拟合）
    num_classes = 2
    vocab_size = 10000
    pad_size = 64   # 适当增大到64，看是否可改善信息丢失
    embed = 128     # 先用128，不要过大
    dim_model = 128
    hidden = 256
    num_encoder = 2  # 先用2层
    num_head = 4     # 4头
    dropout = 0.3    # 增大dropout
    # 训练超参数
    batch_size = 16  # 适当增大
    num_epochs = 150  # 先训练20轮，配合 early stopping
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
