import os

import matplotlib.pyplot as plt
import torch
from torchtext.data.utils import get_tokenizer

from dataloader import get_dataloaders
from inference import translate_sentence
from seq2seq import Encoder, Attention, Decoder, Seq2Seq
from train import init_weights, train_model


def main(train_flag=True):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 1. 获取 DataLoader 和词表
    train_dataloader, val_dataloader, en_vocab, zh_vocab = get_dataloaders(
        batch_size=32,
        test_size=0.1,
        file_path='data/cmn.txt'
    )

    print(f'英文词汇表大小：{len(en_vocab)}')
    print(f'中文词汇表大小：{len(zh_vocab)}')

    # 2. 准备模型
    INPUT_DIM = len(en_vocab)  # 英文词表大小
    OUTPUT_DIM = len(zh_vocab)  # 中文词表大小
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # 初始化编码器、注意力机制、解码器和 Seq2Seq 模型
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attn = Attention(HID_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.apply(init_weights)

    # 3. 训练或加载已有模型
    save_path = 'best_model.pt'
    if train_flag:
        print("[INFO] 开始训练...")
        train_losses, val_losses = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            n_epochs=20,  # 可以根据需要调整
            clip=1,
            lr=1e-3,
            save_path=save_path
        )
        print("[INFO] 训练完成。")

        # 绘制训练和验证损失
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label="Validation Loss")
        plt.title("Train and valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.show()
    else:
        # 如果不训练，且本地已有 best_model.pt，直接加载
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"[INFO] 已加载模型权重自 {save_path}")
        else:
            raise FileNotFoundError(f"模型文件 {save_path} 未找到。请先训练模型或提供正确的路径。")

    # 4. 推理测试
    # 确保加载的是最优模型
    if train_flag and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"[INFO] 已加载最优模型权重自 {save_path}")

    tokenizer_en = get_tokenizer('basic_english')

    print("\n[INFO] 进入交互式翻译模式。输入 'quit' 退出。")
    while True:
        input_sentence = input("请输入英文句子（输入 'quit' 退出）：")
        if input_sentence.lower() == 'quit':
            break
        translation = translate_sentence(
            sentence=input_sentence,
            model=model,
            en_vocab=en_vocab,
            zh_vocab=zh_vocab,
            tokenizer_en=tokenizer_en,
            device=device
        )
        print(f"中文翻译: {translation}\n")


if __name__ == "__main__":
    """
    默认执行：先训练，后推理。
    如果不想训练，只想推理，可以调用 main(train_flag=False)。
    """
    main(train_flag=True)
