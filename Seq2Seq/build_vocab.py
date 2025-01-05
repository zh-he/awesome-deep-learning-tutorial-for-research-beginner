import os
import pickle

from utils import (
    build_vocab_from_dataset,
    TranslationDataset
)

# =============== 配置部分 ===============
max_length = 50
min_count = 2
train_src_file = os.path.join('.', 'train', 'news-commentary-v13.zh-en.en')
train_trg_file = os.path.join('.', 'train', 'news-commentary-v13.zh-en.zh')

src_vocab_path = './src_vocab.pkl'
trg_vocab_path = './trg_vocab.pkl'
# =======================================

def main():
    # 1) 构建原始训练数据集
    train_dataset_raw = TranslationDataset(
        src_file=train_src_file,
        trg_file=train_trg_file,
        max_length=max_length
    )

    # 2) 基于训练集构建词表 (源 & 目标)
    src_vocab, trg_vocab = build_vocab_from_dataset(train_dataset_raw, min_count=min_count)

    # 3) 分别保存到文件
    with open(src_vocab_path, 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(trg_vocab_path, 'wb') as f:
        pickle.dump(trg_vocab, f)

    print(f"词汇表已构建并保存至: {src_vocab_path} 和 {trg_vocab_path}")

if __name__ == '__main__':
    main()
