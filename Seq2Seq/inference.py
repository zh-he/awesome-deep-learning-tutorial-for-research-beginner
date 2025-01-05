import argparse

import torch

from seq2seq import Encoder, Attention, Decoder, Seq2Seq
from utils import tokenize, normalize_sentence, PAD_token, UNK_token, SOS_token, EOS_token


def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    tokens = tokenize(normalize_sentence(sentence))
    # 添加 SOS 和 EOS
    src_indices = [SOS_token] + [src_vocab.word2index.get(word, UNK_token) for word in tokens] + [EOS_token]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    src_length = torch.tensor([len(src_indices)]).to(device)  # [1]

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_length)

    trg_indices = [SOS_token]
    mask = model.create_mask(src_tensor)  # [1, src_len]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)  # [1]

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs, mask)

        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == EOS_token:
            break

    # 去掉最前面的 SOS 和最后面的 EOS
    trg_tokens = [trg_vocab.index2word.get(idx, 'UNK') for idx in trg_indices[1:-1]]

    return trg_tokens


def display_translation(sentence, translation):
    print(f'Source: {sentence}')
    print(f'Translation: {" ".join(translation)}')


def load_vocabulary(src_vocab_path='src_vocab.pkl', trg_vocab_path='trg_vocab.pkl'):
    """
    加载预先保存的词汇表。
    """
    import pickle

    with open(src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)

    with open(trg_vocab_path, 'rb') as f:
        trg_vocab = pickle.load(f)

    return src_vocab, trg_vocab


def load_model(model_path, src_vocab, trg_vocab, device, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.5):
    INPUT_DIM = src_vocab.n_words
    OUTPUT_DIM = trg_vocab.n_words
    ENC_EMB_DIM = emb_dim
    DEC_EMB_DIM = emb_dim
    HID_DIM = hid_dim
    N_LAYERS = n_layers
    ENC_DROPOUT = dropout
    DEC_DROPOUT = dropout

    enc = Encoder(
        input_dim=INPUT_DIM,
        emb_dim=ENC_EMB_DIM,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        dropout=ENC_DROPOUT
    )
    attn = Attention(hid_dim=HID_DIM)
    dec = Decoder(
        output_dim=OUTPUT_DIM,
        emb_dim=DEC_EMB_DIM,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        attention=attn
    )

    model = Seq2Seq(enc, dec, src_pad_idx=PAD_token, device=device).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Translate a sentence.")
    parser.add_argument('--model_path', type=str, default='best-model.pt', help='Path to the trained model.')
    parser.add_argument('--src_vocab_path', type=str, default='src_vocab.pkl', help='Path to the source vocabulary.')
    parser.add_argument('--trg_vocab_path', type=str, default='trg_vocab.pkl', help='Path to the target vocabulary.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: cuda or cpu.')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 加载词汇表
    print("Loading vocabulary...")
    src_vocab, trg_vocab = load_vocabulary(args.src_vocab_path, args.trg_vocab_path)

    # 加载模型
    print("Loading model...")
    model = load_model(args.model_path, src_vocab, trg_vocab, device)

    # 开始翻译
    while True:
        sentence = input("Enter a sentence to translate (or 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        translation = translate_sentence(sentence, src_vocab, trg_vocab, model, device)
        display_translation(sentence, translation)


if __name__ == '__main__':
    main()
