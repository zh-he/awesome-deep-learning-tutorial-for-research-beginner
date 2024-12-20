import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import ResNet


# 数据读取和预处理
def get_data_loaders(data_dir, batch_size=64, valid_ratio=0.2):
    """
    获取训练和验证的数据加载器
    :param data_dir: 数据集根目录，包含 train 和 val 文件夹
    :param batch_size: 批大小
    :param valid_ratio: 验证集比例
    :return: 训练集和验证集的 DataLoader
    """
    # 定义图像预处理
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet的均值
                             [0.229, 0.224, 0.225])  # ImageNet的标准差
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 使用 ImageFolder 加载数据
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train', 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test', 'test'), transform=transform_val)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


# 训练函数
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch}] Training", leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# 验证函数
def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Epoch [{epoch}] Validation", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# 主训练流程
def main():
    # 配置参数
    data_dir = './dogs-vs-cats'  # 数据集路径
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.001
    num_classes = 2
    input_channels = 3
    num_channels = 64  # 初始通道数
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)

    # 初始化模型
    model = ResNet.SimpleResNet(input_channels=input_channels, num_channels=num_channels, num_classes=num_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_per_epoch = []
    train_acc_per_epoch = []
    var_loss_per_epoch = []
    var_acc_per_epoch = []

    # 训练和验证
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, device, val_loader, criterion, epoch)
        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(train_acc)
        var_loss_per_epoch.append(val_loss)
        var_acc_per_epoch.append(val_acc)

        # 绘制四幅图
    epoch_list = list(range(1, num_epochs + 1))
    plt.figure(figsize=(12, 10))

    # 1. 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(epoch_list, train_loss_per_epoch, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # 2. 训练准确率
    plt.subplot(2, 2, 2)
    plt.plot(epoch_list, train_acc_per_epoch, label='Train Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    # 3. 验证损失
    plt.subplot(2, 2, 3)
    plt.plot(epoch_list, var_loss_per_epoch, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # 4. 验证准确率
    plt.subplot(2, 2, 4)
    plt.plot(epoch_list,  var_acc_per_epoch, label='Validation Accuracy', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':
    main()
