import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import load_data, build_vocab, TranslationDataset, collate_fn, PAD_token
from models.seq2seq import Encoder, Attention, Decoder, Seq2Seq
from train import train_seq2seq

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model for Chinese-English translation.")
    parser.add_argument('--train_src', type=str, default='data/train.zh', help='Path to training source data (Chinese).')
    parser.add_argument('--train_trg', type=str, default='data/train.en', help='Path to training target data (English).')
    parser.add_argument('--valid_src', type=str, default='data/valid.zh', help='Path to validation source data (Chinese).')
    parser.add_argument('--valid_trg', type=str, default='data/valid.en', help='Path to validation target data (English).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension.')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden dimension.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--min_count', type=int, default=2, help='Minimum word count to include in vocabulary.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    train_src, train_trg = load_data(args.train_src, args.train_trg)
    valid_src, valid_trg = load_data(args.valid_src, args.valid_trg)

    # 构建词汇表
    src_vocab, trg_vocab = build_vocab(train_src, train_trg, min_count=args.min_count)
    print(f"Source vocabulary size: {src_vocab.n_words}")
    print(f"Target vocabulary size: {trg_vocab.n_words}")

    # 创建数据集
    train_dataset = TranslationDataset(train_src, train_trg, src_vocab, trg_vocab, max_length=50)
    valid_dataset = TranslationDataset(valid_src, valid_trg, src_vocab, trg_vocab, max_length=50)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 初始化模型
    enc = Encoder(input_dim=src_vocab.n_words, emb_dim=args.emb_dim, hid_dim=args.hid_dim, n_layers=args.n_layers, dropout=args.dropout)
    attn = Attention(hid_dim=args.hid_dim)
    dec = Decoder(output_dim=trg_vocab.n_words, emb_dim=args.emb_dim, hid_dim=args.hid_dim, n_layers=args.n_layers, dropout=args.dropout, attention=attn)

    model = Seq2Seq(enc, dec, src_pad_idx=PAD_token, device=device).to(device)

    # 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    # 开始训练
    train_seq2seq(model, train_loader, valid_loader, optimizer, criterion, args.n_epochs, args.clip, device)

if __name__ == '__main__':
    main()
