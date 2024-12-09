import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # Sigmoid 的导数，直接使用激活值


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # ReLU 的导数


def binary_cross_entropy(y_predict, y_true):
    epsilon = 1e-10
    return -np.mean(y_true * np.log(y_predict + epsilon) + (1 - y_true) * np.log(1 - y_predict + epsilon))


def bce_derivative(y_predict, y_true):
    epsilon = 1e-10
    return -(y_true / (y_predict + epsilon)) + ((1 - y_true) / (1 - y_predict + epsilon))


class MLP:
    def __init__(self, hidden_nums, learning_rate, inputs, y, epoch):
        self.weight = {}
        self.bias = {}
        self.hidden_input = {}
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.y = y
        self.epoch = epoch
        self.loss = []
        self.accuracy = []
        # 初始化权重和偏置
        for i in range(len(hidden_nums)):
            if i == 0:  # 输入层到第一个隐藏层
                self.weight[f"w{i}"] = np.random.randn(inputs.shape[1], hidden_nums[i])
            else:  # 隐藏层到隐藏层，或隐藏层到输出层
                self.weight[f"w{i}"] = np.random.randn(hidden_nums[i - 1], hidden_nums[i])
            self.bias[f"b{i}"] = np.zeros((1, hidden_nums[i]))

    def forward(self, x, hidden_nums):
        for i in range(len(hidden_nums)):
            z = np.dot(x, self.weight[f"w{i}"]) + self.bias[f"b{i}"]
            if i != len(hidden_nums) - 1:  # 隐藏层，使用 ReLU
                self.hidden_input[f"h{i}"] = relu(z)
                x = self.hidden_input[f"h{i}"]
            else:  # 输出层，使用 Sigmoid
                self.output = sigmoid(z)

    def backward(self, x, y, hidden_nums):
        delta = None
        for i in range(len(hidden_nums) - 1, -1, -1):  # 从输出层到隐藏层反向传播
            if i == len(hidden_nums) - 1:  # 输出层
                delta = bce_derivative(self.output, y) * sigmoid_derivative(self.output)
                grad_w = np.dot(self.hidden_input[f"h{i - 1}"].T, delta)
            else:  # 隐藏层

                if i > 0:
                    delta = np.dot(delta, self.weight[f"w{i + 1}"].T) * relu_derivative(
                        np.dot(self.hidden_input[f"h{i - 1}"], self.weight[f"w{i}"]) + self.bias[f"b{i}"])
                    grad_w = np.dot(self.hidden_input[f"h{i - 1}"].T, delta)
                else:
                    delta = np.dot(delta, self.weight[f"w{i + 1}"].T) * relu_derivative(
                        np.dot(x, self.weight[f"w{i}"]) + self.bias[f"b{i}"])
                    grad_w = np.dot(x.T, delta)

            # 更新权重和偏置
            self.weight[f"w{i}"] -= self.learning_rate * grad_w
            self.bias[f"b{i}"] -= self.learning_rate * delta

    def train(self, hidden_nums):
        """
        使用 SGD 更新
        """
        for epoch in range(self.epoch):
            for i in range(self.inputs.shape[0]):  # 遍历每个样本
                x = self.inputs[i:i + 1]  # 当前样本，形状 (1, 特征数)
                y = self.y[i:i + 1]  # 当前标签，形状 (1, 输出数)

                # 前向传播
                self.forward(x, hidden_nums)

                # 反向传播
                self.backward(x, y, hidden_nums)

            # 每个 epoch 的损失输出
            predictions = []
            for i in range(self.inputs.shape[0]):
                self.forward(self.inputs[i:i + 1], hidden_nums)
                predictions.append(self.output)
            predictions = np.vstack(predictions)
            loss = binary_cross_entropy(predictions, self.y)

            # 计算准确率
            predicted_classes = np.where(predictions > 0.5, 1, 0)

            accuracy = np.mean(predicted_classes == self.y)
            self.loss.append(loss)
            self.accuracy.append(accuracy)
            print(f"Epoch {epoch + 1}/{self.epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


# 示例运行
np.random.seed(42)

# 数据归一化
input_data = np.random.randn(1000, 5)  # 1000 个样本，5 个特征
# 二分类标签
labels = ((input_data[:, 0] > 0) & (input_data[:, 1] > 0.5) & (input_data[:, 2] < 0)).astype(int).reshape(-1,
                                                                                                          1)  # 每个样本的目标值

# 构建模型
mlp = MLP(hidden_nums=[32, 16, 1], learning_rate=0.001, inputs=input_data, y=labels, epoch=100)

# 训练模型
mlp.train(hidden_nums=[32, 16, 1])

import matplotlib.pyplot as plt

plt.plot(mlp.loss, label="loss")
plt.plot(mlp.accuracy, label="accuracy")
plt.legend()
plt.grid()
plt.show()
