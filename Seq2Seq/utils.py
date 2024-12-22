import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
import os
import numpy as np
import torch.nn as nn

# 定义特殊符号
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

# 下载nltk数据
nltk.download('punkt')

def normalize_sentence(sentence):
    """
    清洗文本，去除多余的空格和标点。
    """
    sentence = sentence.lower().strip()
    # 保留中文和英文字符以及基本标点
    sentence = re.sub(r"([.!?])", r" \1", sentence)
    sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
    return sentence

def tokenize(sentence):
    return word_tokenize(sentence)

class Vocabulary:
    def __init__(self):
        self.word2index = {"SOS": SOS_token, "EOS": EOS_token, "UNK": UNK_token, "PAD": PAD_token}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK", PAD_token: "PAD"}
        self.n_words = 4  # 初始词汇量包括特殊符号

    def add_sentence(self, sentence):
        for word in tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def load_data(src_path, trg_path, max_length=50):
    """
    加载源语言和目标语言的句子。
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src_sentences = f.read().strip().split('\n')

    with open(trg_path, 'r', encoding='utf-8') as f:
        trg_sentences = f.read().strip().split('\n')

    # 确保源和目标句子数量一致
    assert len(src_sentences) == len(trg_sentences), "源语言和目标语言句子数量不匹配"

    # 清洗和分词
    src_sentences = [normalize_sentence(sentence) for sentence in src_sentences]
    trg_sentences = [normalize_sentence(sentence) for sentence in trg_sentences]

    # 过滤过长的句子
    filtered = []
    for src, trg in zip(src_sentences, trg_sentences):
        if len(tokenize(src)) < max_length and len(tokenize(trg)) < max_length:
            filtered.append((src, trg))

    src_sentences, trg_sentences = zip(*filtered)
    return src_sentences, trg_sentences

def build_vocab(src_sentences, trg_sentences, min_count=2):
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()

    for sentence in src_sentences:
        src_vocab.add_sentence(sentence)

    for sentence in trg_sentences:
        trg_vocab.add_sentence(sentence)

    # 过滤低频词
    src_vocab.word2index = {word: idx for word, idx in src_vocab.word2index.items() if src_vocab.word2count.get(word, 0) >= min_count}
    trg_vocab.word2index = {word: idx for word, idx in trg_vocab.word2index.items() if trg_vocab.word2count.get(word, 0) >= min_count}

    # 重建 index2word 和 n_words
    src_vocab.index2word = {idx: word for word, idx in src_vocab.word2index.items()}
    trg_vocab.index2word = {idx: word for word, idx in trg_vocab.word2index.items()}
    src_vocab.n_words = len(src_vocab.word2index)
    trg_vocab.n_words = len(trg_vocab.word2index)

    return src_vocab, trg_vocab

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, max_length=50):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]

        src_indices = [self.src_vocab.word2index.get(word, UNK_token) for word in tokenize(src_sentence)]
        trg_indices = [self.trg_vocab.word2index.get(word, UNK_token) for word in tokenize(trg_sentence)]

        # 添加 SOS 和 EOS
        src_indices = [SOS_token] + src_indices + [EOS_token]
        trg_indices = [SOS_token] + trg_indices + [EOS_token]

        # 截断
        src_indices = src_indices[:self.max_length]
        trg_indices = trg_indices[:self.max_length]

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    """
    填充批次中的序列，使它们具有相同的长度。
    """
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(seq) for seq in src_batch]
    trg_lengths = [len(seq) for seq in trg_batch]

    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_token, batch_first=True)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_token, batch_first=True)

    return src_padded, trg_padded, src_lengths, trg_lengths
