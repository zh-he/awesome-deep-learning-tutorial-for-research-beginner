import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 输出: 32 x 28 x 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 32 x 14 x 14

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输出: 64 x 14 x 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 输出: 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 数据加载与预处理
def get_data_loaders(batch_size=64):
    """
    获取训练和测试的数据加载器。

    参数:
    - batch_size (int): 每个批次的样本数。

    返回:
    - train_loader (DataLoader): 训练集加载器。
    - test_loader (DataLoader): 测试集加载器。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
    ])

    # 加载训练集
    train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 加载测试集
    test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# 训练函数
def train(model, device, train_loader, criterion, optimizer, epoch):
    """
    执行一个训练周期。

    参数:
    - model (nn.Module): 神经网络模型。
    - device (torch.device): 计算设备（CPU 或 GPU）。
    - train_loader (DataLoader): 训练集加载器。
    - criterion (nn.Module): 损失函数。
    - optimizer (torch.optim.Optimizer): 优化器。
    - epoch (int): 当前的训练周期数。

    返回:
    - epoch_loss (float): 当前周期的平均损失。
    - epoch_acc (float): 当前周期的平均准确率。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Train - Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# 验证函数
def evaluate(model, device, test_loader, criterion, epoch):
    """
    在验证集上评估模型性能。

    参数:
    - model (nn.Module): 神经网络模型。
    - device (torch.device): 计算设备（CPU 或 GPU）。
    - test_loader (DataLoader): 测试集加载器。
    - criterion (nn.Module): 损失函数。
    - epoch (int): 当前的训练周期数。

    返回:
    - epoch_loss (float): 当前周期的平均损失。
    - epoch_acc (float): 当前周期的平均准确率。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Validation - Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# 评估函数
def model_test(model, device, test_loader):
    """
    在测试集上评估模型的最终性能。

    参数:
    - model (nn.Module): 神经网络模型。
    - device (torch.device): 计算设备（CPU 或 GPU）。
    - test_loader (DataLoader): 测试集加载器。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")


# 绘制结果曲线
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    绘制训练和验证过程中的损失与准确率曲线。

    参数:
    - train_losses (list of float): 训练集每个周期的损失。
    - train_accuracies (list of float): 训练集每个周期的准确率。
    - val_losses (list of float): 验证集每个周期的损失。
    - val_accuracies (list of float): 验证集每个周期的准确率。
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 主训练流程
def main():
    # 配置参数
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)

    # 初始化模型、损失函数和优化器
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练和验证损失与准确率
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证
        val_loss, val_acc = evaluate(model, device, test_loader, criterion, epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    # 绘制结果
    plot_results(train_losses, train_accuracies, val_losses, val_accuracies)

    # 评估模型
    model_test(model, device, test_loader)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cnn_mnist.pth')
    print("Model saved to 'models/cnn_mnist.pth'")


# 执行主函数
if __name__ == '__main__':
    main()
