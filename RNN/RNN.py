import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# ===== 1. 数据准备 =====
# 中国自1978-2023年的GDP（四舍五入保留两位小数）
gdp_data = np.array([
    0.15, 0.18, 0.19, 0.2, 0.21, 0.23, 0.26, 0.31, 0.3, 0.27, 0.31, 0.35, 0.36, 0.38, 0.43, 0.44, 0.56, 0.73,
    0.86, 0.96, 1.03, 1.09, 1.21, 1.34, 1.47, 1.66, 1.96, 2.29, 2.75, 3.55, 4.59, 5.10, 6.09, 7.55, 8.53, 9.57,
    10.48, 11.06, 11.23, 12.31, 13.89, 14.28, 14.69, 17.82, 17.88, 17.79
], dtype=np.float32)
year = np.arange(1978, 2024)


# 使用 MinMaxScaler 对数据进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
gdp_data_scaled = scaler.fit_transform(gdp_data.reshape(-1, 1)).flatten()


# 创建滑动窗口函数（基于归一化数据）
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


seq_length = 10
X, y = create_sequences(gdp_data_scaled, seq_length)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)  # 80% 数据做训练
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# ===== 2. 定义 RNN 模型 =====
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x.shape = (batch_size, seq_length)
        # 需要添加特征维度 => (batch_size, seq_length, input_size)
        out, _ = self.rnn(x.unsqueeze(-1))
        # 取最后时间步的 hidden state
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 8
output_size = 1

model = SimpleRNN(input_size, hidden_size, output_size)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# ===== 3. 训练模型 =====
num_epochs = 500
loss_list = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

# 绘制训练 Loss 曲线
plt.figure()
plt.plot(loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# ===== 4. 测试模型 =====
model.eval()
with torch.no_grad():
    train_preds = model(X_train).squeeze()
    test_preds = model(X_test).squeeze()

# 反归一化预测结果
train_preds_inv = scaler.inverse_transform(train_preds.numpy().reshape(-1, 1)).flatten()
y_train_inv = scaler.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten()

test_preds_inv = scaler.inverse_transform(test_preds.numpy().reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# ===== 5. 可视化结果 =====
# （1）可视化全部数据（实际 & 预测）
plt.figure(figsize=(6, 4))

# 实际的全部 GDP
plt.plot(year, gdp_data, label='Actual GDP', color='blue')

# 模型对训练集的预测（可视化时要对齐数据点）
train_range = np.arange(1978 + seq_length, 1978+ seq_length + len(train_preds_inv))
plt.plot(train_range, train_preds_inv, label='Train Predicted', color='green')

# 模型对测试集的预测（同理对齐数据点）
test_range = np.arange(1978 + seq_length + len(train_preds_inv),
                       1978 + seq_length + len(train_preds_inv) + len(test_preds_inv))
plt.plot(test_range, test_preds_inv, label='Test Predicted', color='red')

plt.title("GDP Prediction (Train & Test)")
plt.xlabel("Data Index")
plt.ylabel("GDP")
plt.legend()
plt.show()
