# -*- coding: utf-8 -*-
"""
main.py
程序入口
"""
from config import Config
from dataset import create_dataloaders
from model import TransformerClassifier
from trainer import Trainer
from utils import set_seed

# from utils import set_seed

def main():
    # 1. 初始化配置
    config = Config()

    # 如果需要固定随机种子，可取消注释
    set_seed(42)

    # 2. 构建DataLoader，得到训练和验证
    train_loader, val_loader, word2idx = create_dataloaders(config)

    print(f"实际词表大小: {config.vocab_size}")

    # 3. 初始化模型
    model = TransformerClassifier(config)

    # 4. 初始化Trainer并开始训练
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # 5. 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
