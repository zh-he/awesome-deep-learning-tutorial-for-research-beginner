import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # Sigmoid 的导数，直接使用激活值


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)  # ReLU 的导数


def MSE(y_true, y_predict):
    return np.mean((y_true - y_predict) ** 2) / 2


class MLP:
    def __init__(self, hidden_nums, learning_rate, inputs, y, epoch):
        self.weight = {}
        self.bias = {}
        self.hidden_input = {}
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.y = y
        self.epoch = epoch

        # 初始化权重和偏置
        for i in range(len(hidden_nums)):
            if i == 0:  # 输入层到第一个隐藏层
                self.weight[f"w{i}"] = np.random.randn(inputs.shape[1], hidden_nums[i])
                self.bias[f"b{i}"] = np.random.randn(1, hidden_nums[i])
            else:  # 隐藏层到隐藏层，或隐藏层到输出层
                self.weight[f"w{i}"] = np.random.randn(hidden_nums[i - 1], hidden_nums[i])
                self.bias[f"b{i}"] = np.random.randn(1, hidden_nums[i])

    def forward(self, x, hidden_nums):
        for i in range(len(hidden_nums)):
            if i != len(hidden_nums) - 1:  # 隐藏层，使用 Sigmoid
                z = np.dot(x, self.weight[f"w{i}"]) + self.bias[f"b{i}"]
                self.hidden_input[f"h{i}"] = sigmoid(z)
                x = self.hidden_input[f"h{i}"]
            else:  # 输出层，使用 ReLU
                z = np.dot(x, self.weight[f"w{i}"]) + self.bias[f"b{i}"]
                self.output = relu(z)

    def backward(self, x, y, hidden_nums):
        delta = None
        for i in range(len(hidden_nums) - 1, -1, -1):  # 从输出层到隐藏层反向传播
            if i == len(hidden_nums) - 1:  # 输出层
                delta = (self.output - y) * relu_derivative(self.output)  # 损失对 ReLU 的导数
                grad_w = np.dot(self.hidden_input[f"h{i - 1}"].T, delta)
            else:  # 隐藏层
                delta = np.dot(delta, self.weight[f"w{i + 1}"].T) * sigmoid_derivative(self.hidden_input[f"h{i}"])
                if i > 0:
                    grad_w = np.dot(self.hidden_input[f"h{i - 1}"].T, delta)
                else:
                    grad_w = np.dot(x.T, delta)

            # 更新权重和偏置
            self.weight[f"w{i}"] -= self.learning_rate * grad_w
            self.bias[f"b{i}"] -= self.learning_rate * delta

    def train(self, hidden_nums):
        """
        逐样本使用 SGD 更新
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
            loss = MSE(self.y, predictions)
            # 计算准确率
            predicted_classes = (predictions > 0.7).astype(int)  # 假设二分类任务，阈值为 0.5
            true_classes = (self.y > 0.7).astype(int)  # 将真实标签转换为二分类
            accuracy = np.mean(predicted_classes == true_classes)  # 比较预测与真实值

            print(f"Epoch {epoch + 1}/{self.epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


# 示例运行
np.random.seed(88)
input_data = np.random.rand(1000, 5) * 5 # 10 个样本，5 个特征
labels = np.random.rand(1000, 1)  # 每个样本的目标值
mlp = MLP(hidden_nums=[6, 4, 2, 1], learning_rate=0.01, inputs=input_data, y=labels, epoch=50)

# 训练
mlp.train(hidden_nums=[6, 4, 2, 1])
