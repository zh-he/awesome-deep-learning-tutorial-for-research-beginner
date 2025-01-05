import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os

from utils import (
    collate_fn,
    PAD_token,
    build_vocab_from_dataset,
    TranslationDataset
)
from seq2seq import Encoder, Attention, Decoder, Seq2Seq
from train import train_seq2seq

# ---------------------
# 超参数设置
# ---------------------
batch_size = 128
emb_dim = 128
hid_dim = 128
n_layers = 1

n_epochs = 10
clip = 1.0
learning_rate = 0.001
max_length = 50

# ---------------------
# 数据集路径 (根据你本地目录结构进行修改)
# ---------------------
train_src_file = os.path.join('.', 'train', 'news-commentary-v13.zh-en.en')
train_trg_file = os.path.join('.', 'train', 'news-commentary-v13.zh-en.zh')

valid_src_file = os.path.join('.', 'test', 'newstest2017.tc.en')
valid_trg_file = os.path.join('.', 'test', 'newstest2017.tc.zh')

# ---------------------
# 已经构建好的词汇表路径
# ---------------------
src_vocab_path = './src_vocab.pkl'
trg_vocab_path = './trg_vocab.pkl'

# ---------------------
# 0) 加载词汇表
# ---------------------
def load_vocabulary(src_path, trg_path):
    with open(src_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(trg_path, 'rb') as f:
        trg_vocab = pickle.load(f)
    return src_vocab, trg_vocab

# ---------------------
# 1) 构建训练集 & 验证集
# ---------------------
train_dataset_raw = TranslationDataset(
    src_file=train_src_file,
    trg_file=train_trg_file,
    max_length=max_length
)

valid_dataset_raw = TranslationDataset(
    src_file=valid_src_file,
    trg_file=valid_trg_file,
    max_length=max_length
)

# ---------------------
# 2) 从文件加载词表
# ---------------------
src_vocab, trg_vocab = load_vocabulary(src_vocab_path, trg_vocab_path)

print(f"源词表大小: {src_vocab.n_words}, 目标词表大小: {trg_vocab.n_words}")

# ---------------------
# 3) 重新构建 “可训练” 的 Dataset (包含数值化处理)
# ---------------------
train_dataset = []
for i in range(len(train_dataset_raw)):
    src_line, trg_line = train_dataset_raw[i]
    train_dataset.append((src_line, trg_line))

valid_dataset = []
for i in range(len(valid_dataset_raw)):
    src_line, trg_line = valid_dataset_raw[i]
    valid_dataset.append((src_line, trg_line))

class NumberizedTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, text_pairs, src_vocab, trg_vocab, max_length=50):
        self.text_pairs = text_pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        src_sentence, trg_sentence = self.text_pairs[idx]
        from utils import tokenize, SOS_token, EOS_token, UNK_token

        src_tokens = tokenize(src_sentence)
        trg_tokens = tokenize(trg_sentence)

        src_indices = [SOS_token] + [self.src_vocab.word2index.get(t, UNK_token) for t in src_tokens] + [EOS_token]
        trg_indices = [SOS_token] + [self.trg_vocab.word2index.get(t, UNK_token) for t in trg_tokens] + [EOS_token]

        # 截断
        src_indices = src_indices[:self.max_length]
        trg_indices = trg_indices[:self.max_length]

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

train_numberized_dataset = NumberizedTranslationDataset(train_dataset, src_vocab, trg_vocab, max_length)
valid_numberized_dataset = NumberizedTranslationDataset(valid_dataset, src_vocab, trg_vocab, max_length)

# ---------------------
# 4) 构建 DataLoader
# ---------------------
train_loader = DataLoader(
    train_numberized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, src_vocab, trg_vocab)
)

valid_loader = DataLoader(
    valid_numberized_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, src_vocab, trg_vocab)
)

# ---------------------
# 5) 初始化模型
# ---------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

INPUT_DIM = src_vocab.n_words
OUTPUT_DIM = trg_vocab.n_words

enc = Encoder(
    input_dim=INPUT_DIM,
    emb_dim=emb_dim,
    hid_dim=hid_dim,
    n_layers=n_layers,

)
attn = Attention(hid_dim=hid_dim)
dec = Decoder(
    output_dim=OUTPUT_DIM,
    emb_dim=emb_dim,
    hid_dim=hid_dim,
    n_layers=n_layers,
    attention=attn
)

model = Seq2Seq(
    encoder=enc,
    decoder=dec,
    src_pad_idx=PAD_token,
    device=device
).to(device)

# ---------------------
# 6) 优化器 & 损失函数
# ---------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

# ---------------------
# 7) 训练与验证
# ---------------------
if __name__ == '__main__':
    train_seq2seq(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        N_EPOCHS=n_epochs,
        CLIP=clip,
        device=device
    )
