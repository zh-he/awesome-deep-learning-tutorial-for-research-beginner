import torch
import torch.nn as nn
import torch.nn.functional as F

""""
    Author: Zhihai He
    Date: 2024/12/18
    Data Source: 链接：https://pan.baidu.com/s/1l1AnBgkAAEhh0vI5_loWKw 提取码：2xq4 blog:https://blog.csdn.net/dddccc1234/article/details/122622182
    Description: 利用ResNet实现猫狗分类
"""

# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) if strides != 1 or input_channels != num_channels else None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = self.conv3(x) if self.conv3 else x
        return F.relu(x + y)


class SimpleResNet(nn.Module):
    def __init__(self, input_channels, num_channels, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(num_channels, num_channels, blocks=2, stride=1),  # layer1
            self._make_layer(num_channels, num_channels * 2, blocks=2, stride=2),  # layer2
            self._make_layer(num_channels * 2, num_channels * 4, blocks=2, stride=2),  # layer3
            self._make_layer(num_channels * 4, num_channels * 8, blocks=2, stride=2),  # layer4
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(num_channels * 8, num_classes)

    def _make_layer(self, input_channels, num_channels, blocks, stride=2):
        layers = []
        layers.append(Residual(input_channels, num_channels, strides=stride))
        for _ in range(1, blocks):
            layers.append(Residual(num_channels, num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x