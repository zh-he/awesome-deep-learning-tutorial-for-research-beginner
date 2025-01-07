import torch


def translate_sentence(sentence, model, en_vocab, zh_vocab, tokenizer_en, device, max_len=50):
    """
    使用训练好的 Seq2Seq 模型翻译给定的英文句子
    """
    model.eval()

    # 对输入句子进行分词，并添加 <bos> / <eos>
    tokens = tokenizer_en(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    src_indices = [en_vocab.stoi.get(t, en_vocab.stoi['<unk>']) for t in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # 解码器的初始输入是 <bos>
    trg_indices = [zh_vocab.stoi['<bos>']]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == zh_vocab.stoi['<eos>']:
            break

    trg_tokens = [zh_vocab.itos[idx] for idx in trg_indices]
    # 去掉首尾 <bos> <eos>
    return "".join(trg_tokens[1:-1])


def interactive_inference(model, en_vocab, zh_vocab, tokenizer_en, device):
    """
    命令行交互式翻译
    """
    while True:
        input_sentence = input("请输入英文句子（输入 'quit' 退出）：")
        if input_sentence.lower() == 'quit':
            break
        translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en, device)
        print(f"中文翻译: {translation}")
