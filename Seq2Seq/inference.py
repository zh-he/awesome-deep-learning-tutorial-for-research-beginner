# inference.py

import torch
import torch.nn as nn
from models.seq2seq import Encoder, Attention, Decoder, Seq2Seq
from utils import tokenize, normalize_sentence, build_vocab, Vocabulary, PAD_token, UNK_token, SOS_token, EOS_token
import argparse

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    tokens = tokenize(normalize_sentence(sentence))
    tokens = [token for token in tokens]  # 保留原始顺序
    src_indices = [src_vocab.word2index.get(word, UNK_token) for word in tokens]
    src_indices = [SOS_token] + src_indices + [EOS_token]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    src_length = [len(src_indices)]

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)

    trg_indices = [SOS_token]

    mask = model.create_mask(src_tensor)

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)  # [1]

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == EOS_token:
            break

    trg_tokens = [trg_vocab.index2word.get(idx, 'UNK') for idx in trg_indices[1:-1]]  # 去除 SOS 和 EOS

    return trg_tokens

def display_translation(sentence, translation):
    print(f'Source: {sentence}')
    print(f'Translation: {" ".join(translation)}')

def load_vocabulary():
    """
    加载或构建词汇表。
    这里假设词汇表已经通过训练过程构建，并在训练后保存。
    您需要实现词汇表的保存和加载功能。
    """
    # 示例：直接构建词汇表（实际情况应加载已保存的词汇表）
    # 请根据实际情况调整
    from utils import load_data, build_vocab

    # 加载训练数据以重建词汇表
    train_src, train_trg = load_data('data/train.zh', 'data/train.en')
    src_vocab, trg_vocab = build_vocab(train_src, train_trg, min_count=2)
    return src_vocab, trg_vocab

def load_model(model_path, src_vocab, trg_vocab, device):
    INPUT_DIM = src_vocab.n_words
    OUTPUT_DIM = trg_vocab.n_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(input_dim=INPUT_DIM, emb_dim=ENC_EMB_DIM, hid_dim=HID_DIM, n_layers=N_LAYERS, dropout=ENC_DROPOUT)
    attn = Attention(hid_dim=HID_DIM)
    dec = Decoder(output_dim=OUTPUT_DIM, emb_dim=DEC_EMB_DIM, hid_dim=HID_DIM, n_layers=N_LAYERS, dropout=DEC_DROPOUT, attention=attn)

    model = Seq2Seq(enc, dec, src_pad_idx=PAD_token, device=device).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser(description="Translate a Chinese sentence to English.")
    parser.add_argument('--model_path', type=str, default='best-model.pt', help='Path to the trained model.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载词汇表
    src_vocab, trg_vocab = load_vocabulary()

    # 加载模型
    model = load_model(args.model_path, src_vocab, trg_vocab, device)

    # 开始翻译
    while True:
        sentence = input("Enter a Chinese sentence to translate (or 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        translation = translate_sentence(sentence, src_vocab, trg_vocab, model, device)
        display_translation(sentence, translation)

if __name__ == '__main__':
    main()
