# -*- coding: utf-8 -*-
"""
trainer.py
封装训练与验证过程
"""

import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, model, config, train_loader, val_loader):
        """
        :param model: TransformerClassifier 模型
        :param config: Config 实例
        :param train_loader: 训练集 DataLoader
        :param val_loader: 验证集 DataLoader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化器
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.config.device)

        self.train_loss =[]
        self.train_acc = []
        self.var_loss = []
        self.var_acc = []

    def train(self):
        """
        进行完整的训练流程: 多个epoch
        """
        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate()
            self.var_loss.append(val_loss)
            self.var_acc.append(val_acc)
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        self.plot()

    def train_one_epoch(self, epoch):
        """
        单个epoch的训练流程
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 计算accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total
        self.train_loss.append(avg_loss)
        self.train_acc.append(acc)
        print(f"Train - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    def evaluate(self):
        """
        在验证集上评估模型性能(可扩展到测试集)
        :return: (val_loss, val_acc)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total

        return avg_loss, acc


    def plot(self):
        plt.figure()
        plt.plot(range(1, len(self.var_loss) + 1), self.var_loss, marker="o", label="Var Loss")
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, marker="x", label="Train Loss")
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(range(1, len(self.var_acc) + 1), self.var_acc, marker="o", label="Var Accuracy")
        plt.plot(range(1, len(self.train_acc) + 1), self.train_acc, marker="x", label="Train Accuracy")
        plt.grid()
        plt.legend()
        plt.show()
