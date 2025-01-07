# dataset.py
# -*- coding: utf-8 -*-

import collections
import os
import random
import re
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config

UNK, PAD, CLS, SEP = '<UNK>', '<PAD>', '<CLS>', '<SEP>'


def clean_text(text: str) -> str:
    """
    文本清洗：移除特殊字符，统一格式
    """
    text = text.lower()  # 转为小写
    text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
    text = re.sub(r'\s+', ' ', text)  # 连续空白->单空格
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # 保留字母、数字、空格
    return text.strip()


def tokenizer(text: str) -> List[str]:
    """简单空格分词"""
    return text.split()


def build_vocab(texts: List[str], min_freq: int) -> Dict[str, int]:
    """构建词表（不在 text_to_ids 中再次清洗）"""
    counter = collections.Counter()
    for text in texts:
        words = tokenizer(text)  # 此时文本已在 load_data 阶段清洗过
        counter.update(words)

    # 按频率排序，过滤低频词
    vocab_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    vocab = [w for w, cnt in vocab_freq if cnt >= min_freq]

    # 加入特殊token
    vocab = [PAD, UNK, CLS, SEP] + vocab
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx


def text_to_ids(text: str, word2idx: Dict[str, int], pad_size: int) -> Tuple[List[int], int]:
    """
    将文本转换为id序列
    已在load_data时清洗，此处只做切分+截断/填充+映射
    """
    tokens = tokenizer(text)  # 这里不再调用 clean_text

    # 实际序列长度包含 [CLS] 与 [SEP]
    seq_len = len(tokens) + 2

    # 添加特殊 token
    tokens = [CLS] + tokens + [SEP]

    # 截断或填充
    if len(tokens) < pad_size:
        tokens.extend([PAD] * (pad_size - len(tokens)))
    else:
        tokens = tokens[:pad_size - 1] + [SEP]
        seq_len = pad_size

    # 映射
    input_ids = [word2idx.get(token, word2idx[UNK]) for token in tokens]
    return input_ids, seq_len


def load_data(config: Config) -> Tuple[List[str], List[int]]:
    """
    从文件加载数据，并在此处完成一次性的文本清洗
    """
    pos_path = os.path.join(config.data_dir, config.pos_file)
    neg_path = os.path.join(config.data_dir, config.neg_file)

    texts, labels = [], []

    # 读取正面
    with open(pos_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = clean_text(line)  # 在这里清洗
            texts.append(line)
            labels.append(1)

    # 读取负面
    with open(neg_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = clean_text(line)  # 在这里清洗
            texts.append(line)
            labels.append(0)

    return texts, labels


def split_data(
        texts: List[str],
        labels: List[int],
        train_ratio: float,
        val_ratio: float
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    划分数据集，仅训练/验证
    """
    assert len(texts) == len(labels), "texts和labels长度不匹配"
    data = list(zip(texts, labels))
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]

    train_texts, train_labels = zip(*train_data)
    val_texts, val_labels = zip(*val_data)

    return (list(train_texts), list(train_labels)), (list(val_texts), list(val_labels))


class MRDataset(Dataset):
    """
    电影评论数据集
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 word2idx: Dict[str, int],
                 pad_size: int):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.pad_size = pad_size
        self.contents = []
        self._prepare_data()

    def _prepare_data(self):
        for text, label in zip(self.texts, self.labels):
            input_ids, seq_len = text_to_ids(text, self.word2idx, self.pad_size)
            self.contents.append((input_ids, seq_len, label))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        input_ids, seq_len, label = self.contents[idx]
        # attention mask
        attention_mask = [1 if i < seq_len else 0 for i in range(self.pad_size)]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(config: Config):
    """
    创建训练/验证集加载器
    """
    # 1. 加载数据（并在load_data中完成文本清洗）
    texts, labels = load_data(config)

    # 2. 划分数据
    (train_texts, train_labels), (val_texts, val_labels) = split_data(
        texts, labels, config.train_ratio, config.val_ratio
    )

    # 3. 构建词表（仅使用训练集）
    word2idx = build_vocab(train_texts, config.min_freq)
    config.vocab_size = len(word2idx)

    # 4. 构建Dataset
    train_dataset = MRDataset(train_texts, train_labels, word2idx, config.pad_size)
    val_dataset = MRDataset(val_texts, val_labels, word2idx, config.pad_size)

    # 5. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下可设为0，Linux可根据需要指定
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, word2idx
