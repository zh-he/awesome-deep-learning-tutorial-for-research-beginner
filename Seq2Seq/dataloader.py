from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


def read_and_split_raw_data(file_path='data/cmn.txt'):
    """
    读取原始 cmn.txt 文件，并拆分为英文和中文句子，然后保存至本地。
    返回：包含英文句子列表和中文句子列表的 DataFrame。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                english_sentence = parts[0].strip()
                chinese_sentence = parts[1].strip()
                data.append([english_sentence, chinese_sentence])

    df = pd.DataFrame(data, columns=['English', 'Chinese'])

    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['English'].to_csv('data/english_sentences.txt', index=False, header=False)
    df['Chinese'].to_csv('data/chinese_sentences.txt', index=False, header=False)

    return df


def load_sentences(english_path='data/english_sentences.txt',
                   chinese_path='data/chinese_sentences.txt'):
    """
    从文本文件中读取所有的英文和中文句子，返回列表。
    """
    with open(english_path, 'r', encoding='utf-8') as f:
        english_sentences = [line.strip() for line in f]

    with open(chinese_path, 'r', encoding='utf-8') as f:
        chinese_sentences = [line.strip() for line in f]

    return english_sentences, chinese_sentences


def tokenizer_zh(text):
    """
    中文分词器：此处简单地将每个汉字视为一个 token。
    你也可以使用更复杂的中文分词工具。
    """
    return list(text)


def build_vocab(sentences, tokenizer, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
    """
    根据给定的句子列表和分词器构建词汇表。
    """
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)
    vocab = Vocab(counter, specials=specials, specials_first=True)
    return vocab


def process_sentence(sentence, tokenizer, vocab):
    """
    将句子转换为索引序列，并添加 <bos> 和 <eos>
    """
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab[token] for token in tokens]
    return indices


class TranslationDataset(Dataset):
    """
    自定义数据集，用于存储 (src_indices, trg_indices)
    """

    def __init__(self, src_sequences, trg_sequences, complexities):
        assert len(src_sequences) == len(trg_sequences) == len(complexities)
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences
        self.complexities = complexities  # 用于分层划分

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx]), self.complexities[idx]


def collate_fn(batch, pad_idx_src, pad_idx_trg):
    """
    自定义的 collate_fn，用于将批次中的样本进行填充对齐
    """
    src_batch, trg_batch, _ = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=pad_idx_src)  # [src_len, batch_size]
    trg_batch = pad_sequence(trg_batch, padding_value=pad_idx_trg)  # [trg_len, batch_size]
    return src_batch, trg_batch


def get_dataloaders(batch_size=32, test_size=0.1, file_path='data/cmn.txt'):
    """
    读取数据、构建词汇表、生成索引序列，并返回训练和验证的 DataLoader
    """
    df = read_and_split_raw_data(file_path)
    print(df.head())

    # 定义分词器
    tokenizer_en = get_tokenizer('basic_english')

    # 读取英文、中文句子
    english_sentences, chinese_sentences = load_sentences()

    # 构建词汇表
    en_vocab = build_vocab(english_sentences, tokenizer_en)
    zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

    # 转换为索引序列
    en_sequences = [process_sentence(s, tokenizer_en, en_vocab) for s in english_sentences]
    zh_sequences = [process_sentence(s, tokenizer_zh, zh_vocab) for s in chinese_sentences]

    # 定义复杂度指标，例如句子长度（字符数）
    complexities = [len(s) for s in english_sentences]  # 或者其他复杂度指标

    # 创建 Dataset
    dataset = TranslationDataset(en_sequences, zh_sequences, complexities)

    # 随机划分训练集和验证集
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        random_state=42,
        shuffle=True
    )

    # 根据索引获取训练和验证数据
    train_src = [dataset.src_sequences[i] for i in train_indices]
    train_trg = [dataset.trg_sequences[i] for i in train_indices]
    train_complexities = [dataset.complexities[i] for i in train_indices]

    val_src = [dataset.src_sequences[i] for i in val_indices]
    val_trg = [dataset.trg_sequences[i] for i in val_indices]
    val_complexities = [dataset.complexities[i] for i in val_indices]

    # 创建训练和验证的 Dataset
    train_dataset = TranslationDataset(train_src, train_trg, train_complexities)
    val_dataset = TranslationDataset(val_src, val_trg, val_complexities)

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 这里依然需要打乱
        collate_fn=lambda b: collate_fn(b, en_vocab['<pad>'], zh_vocab['<pad>'])
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, en_vocab['<pad>'], zh_vocab['<pad>'])
    )

    return train_dataloader, val_dataloader, en_vocab, zh_vocab
