import pickle
import re

import nltk
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

# 定义特殊符号
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

# 下载 nltk 数据（若已下载可注释掉）
nltk.download('punkt', quiet=True)


def normalize_sentence(sentence):
    """
    清洗文本，去除多余的空格和标点。
    对于中英文混合的句子，保留中文字符、英文字符以及基本标点。
    """
    sentence = sentence.strip()
    # 在中文/英文和标点之间加空格，方便分词
    sentence = re.sub(r'([。！？,.!?])', r' \1 ', sentence)
    # 保留中英文、数字及常用标点，其他替换为空格
    sentence = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5,.!?]+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


def tokenize(sentence):
    """
    将句子分词，支持中英文混合：
    - 中文字符逐字切分
    - 英文部分用简单逻辑拼接再用 nltk.word_tokenize
    """
    tokens = []
    for char in sentence:
        if re.match(r'[\u4e00-\u9fa5]', char):
            tokens.append(char)
        elif re.match(r'[a-zA-Z0-9]', char):
            # 英文或数字部分拼接
            if tokens and re.match(r'[a-zA-Z0-9]', tokens[-1]):
                tokens[-1] += char
            else:
                tokens.append(char)
        elif char in ['.', '!', '?', ',', '。', '？', '！']:
            tokens.append(char)
        else:
            # 其他字符忽略或视为分隔
            tokens.append(' ')

    # 对拼接的英文词汇做进一步的细分
    final_tokens = []
    for tk in tokens:
        tk = tk.strip()
        if re.match(r'[a-zA-Z0-9]+', tk):
            final_tokens.extend(word_tokenize(tk))
        else:
            if tk:
                final_tokens.append(tk)

    # 去除空白 token
    final_tokens = [tk for tk in final_tokens if tk.strip()]

    return final_tokens


class Vocabulary:
    def __init__(self):
        self.word2index = {
            "SOS": 0,
            "EOS": 1,
            "UNK": 2,
            "PAD": 3
        }
        # 在这里也给特殊符号初始化计数为 0
        self.word2count = {
            "SOS": 0,
            "EOS": 0,
            "UNK": 0,
            "PAD": 0
        }
        self.index2word = {
            0: "SOS",
            1: "EOS",
            2: "UNK",
            3: "PAD"
        }
        self.n_words = 4  # 4 个特殊符号

    # 计入了 4 个特殊符号

    def add_sentence(self, sentence):
        words = tokenize(sentence)
        for w in words:
            self.add_word(w)

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
    加载源语言和目标语言的句子并进行基本清理和过滤。
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = f.read().strip().split('\n')

    with open(trg_path, 'r', encoding='utf-8') as f:
        trg_lines = f.read().strip().split('\n')

    assert len(src_lines) == len(trg_lines), "源语言和目标语言行数不匹配"

    # 清洗
    src_lines = [normalize_sentence(s) for s in src_lines]
    trg_lines = [normalize_sentence(s) for s in trg_lines]

    filtered_src = []
    filtered_trg = []
    for s, t in zip(src_lines, trg_lines):
        s_tokens = tokenize(s)
        t_tokens = tokenize(t)
        if len(s_tokens) <= max_length and len(t_tokens) <= max_length:
            filtered_src.append(s)
            filtered_trg.append(t)

    return filtered_src, filtered_trg


def build_vocab(src_sentences, trg_sentences, min_count=2):
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()

    for s in src_sentences:
        src_vocab.add_sentence(s)
    for t in trg_sentences:
        trg_vocab.add_sentence(t)

    # 过滤低频词
    def filter_vocab(old_vocab):
        new_vocab = Vocabulary()
        for word, idx in old_vocab.word2index.items():
            if word in ["SOS", "EOS", "UNK", "PAD"]:
                # 直接加特殊符号
                continue
        # 将已有的计数信息拷贝过来，以便执行频次判断
        for word, count in old_vocab.word2count.items():
            if count >= min_count and word not in ["SOS", "EOS", "UNK", "PAD"]:
                _ = new_vocab.add_word(word)

        # 把特殊词加回去（并保持原始索引）
        # 由于上面 new_vocab 又重新计数，所以这里需要修正
        # 简单处理：把 new_vocab 里的词全部顺延
        offset = new_vocab.n_words
        # 先备份 old -> new 的映射
        old2new = {}
        for w, i in new_vocab.word2index.items():
            old2new[w] = i
        # 重置 new_vocab
        new_vocab_final = Vocabulary()

        # 把四个特殊词重新放进去
        # 但是要保证它们的 index 不变(0,1,2,3)
        # 这里因为写起来比较繁琐，就简单放进去，然后把正常词继续添加
        # 这样做 index 可能并不会 0,1,2,3，但效果上是等价的
        # 如果一定要保持绝对顺序，需要更复杂的写法
        # 在实际场景下，只要 PAD, UNK, EOS, SOS 的数字标记一致即可
        for w in old_vocab.index2word.values():
            # 先跳过
            pass

        # 重新添加过滤后的词
        for w, c in old_vocab.word2count.items():
            if w in ["SOS", "EOS", "UNK", "PAD"]:
                continue
            if c >= min_count:
                new_vocab_final.add_word(w)

        # 此时 new_vocab_final 的 word2index 一定是从4开始的
        # 所以它和最初 old_vocab 里[0,1,2,3]的映射会有差异
        # 这里我们不做特别强行对齐，只要脚本里后续一致使用即可
        return new_vocab_final

    src_vocab_filtered = filter_vocab(src_vocab)
    trg_vocab_filtered = filter_vocab(trg_vocab)

    # 最后还要把 4 个特殊符号加回来
    # 这里简化做法：只要在构造 Vocabulary() 时已经包含，就算是加回了
    # 只需注意：每次 new_vocab 的 n_words 都从 4 开始

    return src_vocab_filtered, trg_vocab_filtered


def build_vocab_from_dataset(dataset, min_count=2):
    """
    给定一个自定义的 TranslationDataset(Dataset)，其中每个元素是 (src_line, trg_line)。
    批量构建 src_vocab & trg_vocab。
    """
    src_sentences = [item[0] for item in dataset]
    trg_sentences = [item[1] for item in dataset]
    return build_vocab(src_sentences, trg_sentences, min_count)


class TranslationDataset(Dataset):
    """
    用于在最初阶段（只做文本过滤，不做数值化）的简单 Dataset：
    每次 __getitem__ 返回 (原始src句子, 原始trg句子)，不做任何额外处理
    """

    def __init__(self, src_file, trg_file, max_length=50):
        self.src_sentences, self.trg_sentences = load_data(src_file, trg_file, max_length)

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.trg_sentences[idx]


def collate_fn(batch, src_vocab, trg_vocab):
    """
    将 [ (src_indices, trg_indices), (src_indices, trg_indices), ... ] 填充到同等长度。
    这里 batch 是已经数值化后的结果 (见 main.py 中 NumberizedTranslationDataset)。
    """
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(seq) for seq in src_batch]
    trg_lengths = [len(seq) for seq in trg_batch]

    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_token, batch_first=True)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_token, batch_first=True)

    return src_padded, trg_padded, src_lengths, trg_lengths


def save_vocab(vocab, filepath):
    """
    将词汇表保存到文件。
    """
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(src_vocab_path, trg_vocab_path):
    """
    从文件加载词汇表。
    """
    with open(src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)

    with open(trg_vocab_path, 'rb') as f:
        trg_vocab = pickle.load(f)

    return src_vocab, trg_vocab
