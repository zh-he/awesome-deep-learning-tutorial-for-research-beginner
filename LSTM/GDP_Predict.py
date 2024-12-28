import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(3407)

# 1. 数据准备
# 读取 GDP 数据
filename = 'gdp_data.xlsx'  # 确保文件扩展名正确
sheet_names = ['China', 'United States']

# 读取每个工作表，并添加一个 'Country' 列
data_frames = []
for sheet in sheet_names:
    df = pd.read_excel(filename, sheet_name=sheet)
    df['Country'] = sheet  # 添加国家名称列
    data_frames.append(df)

# 合并所有工作表的数据
country_data = pd.concat(data_frames, ignore_index=True)
print(country_data)
# 选择需要的列
# 假设每个工作表都有 'Year' 和 'GDP (trillion)' 列
# 如果列名不同，请根据实际情况调整
required_columns = ['Country', 'Year', 'GDP(triliion)']
if not all(col in country_data.columns for col in required_columns):
    raise ValueError(f"Excel 文件缺少必要的列。需要的列: {required_columns}")

country_data = country_data[required_columns]

# 重命名列
country_data.columns = ['Country', 'Year', 'GDP']

# 可视化 GDP 趋势
plt.figure(figsize=(12, 6))
for country in sheet_names:
    country_subset = country_data[country_data['Country'] == country]
    plt.plot(country_subset['Year'], country_subset['GDP'],
             marker='o', label=country)
plt.title('GDP Over Years for China and United States')
plt.xlabel('Year')
plt.ylabel('GDP (Trillion USD)')
plt.legend()
plt.grid()
plt.show()

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
# 需要按国家分别归一化，以避免数据泄漏
country_data['GDP_scaled'] = 0.0  # 初始化列

for country in sheet_names:
    country_subset = country_data[country_data['Country'] == country]
    scaled_values = scaler.fit_transform(country_subset[['GDP']])
    country_data.loc[country_data['Country'] == country, 'GDP_scaled'] = scaled_values

# 可视化归一化后的 GDP
plt.figure(figsize=(12, 6))
for country in sheet_names:
    country_subset = country_data[country_data['Country'] == country]
    plt.plot(country_subset['Year'], country_subset['GDP_scaled'],
             marker='o', label=country)
plt.title('Scaled GDP Over Years for China and United States')
plt.xlabel('Year')
plt.ylabel('Scaled GDP')
plt.legend()
plt.grid()
plt.show()

# 创建序列数据
sequence_length = 5


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# 为每个国家创建序列数据
sequences = {}
for country in sheet_names:
    country_seq = country_data[country_data['Country'] == country]['GDP_scaled'].values
    X, y = create_sequences(country_seq, sequence_length)
    sequences[country] = {'X': X, 'y': y}
    print(f"{country} - 输入序列形状: {X.shape}, 目标序列形状: {y.shape}")


# 创建数据集和数据加载器
class GDPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # LSTM 期望输入形状为 (seq_len, input_size)
        return self.X[idx].unsqueeze(1), self.y[idx]


batch_size = 6
dataloaders = {}
for country in sheet_names:
    dataset = GDPDataset(sequences[country]['X'], sequences[country]['y'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataloaders[country] = dataloader
    print(f"{country} - DataLoader 创建完成，批次数: {len(dataloader)}")


# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义训练参数
num_epochs = 100
learning_rate = 0.001

# 初始化模型、损失函数和优化器
models = {}
criteria = {}
optimizers_dict = {}
train_losses = {country: [] for country in sheet_names}

for country in sheet_names:
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    models[country] = model
    criteria[country] = criterion
    optimizers_dict[country] = optimizer
    print(f"{country} - 模型、损失函数和优化器初始化完成")

# 训练循环
for epoch in range(1, num_epochs + 1):
    for country in sheet_names:
        model = models[country]
        criterion = criteria[country]
        optimizer = optimizers_dict[country]
        dataloader = dataloaders[country]

        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}] {country} Training", leave=False)
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item() * X_batch.size(0)

            # 更新进度条
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses[country].append(epoch_loss)

        # 打印每个 epoch 的损失
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}], {country} Loss: {epoch_loss:.4f}")

# 绘制训练损失
plt.figure(figsize=(12, 6))
for country in sheet_names:
    plt.plot(range(1, num_epochs + 1), train_losses[country], label=f'{country} Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()
plt.show()

# 模型评估
predictions = {country: [] for country in sheet_names}
actual = {country: [] for country in sheet_names}

# 创建测试数据加载器
test_dataloaders = {}
for country in sheet_names:
    test_dataset = GDPDataset(sequences[country]['X'], sequences[country]['y'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_dataloaders[country] = test_loader
    print(f"{country} - 测试 DataLoader 创建完成，样本数: {len(test_loader)}")

# 进行预测
for country in sheet_names:
    model = models[country]
    model.eval()
    test_loader = test_dataloaders[country]

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=f"{country} Evaluating", leave=False):
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            y_pred = y_pred.cpu().numpy()
            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(y_batch.unsqueeze(1).cpu().numpy())

            predictions[country].append(y_pred.flatten()[0])
            actual[country].append(y_true.flatten()[0])

# 打印预测结果
for country in sheet_names:
    print(f"\n{country} - 预测结果与实际值对比:")
    for i in range(len(predictions[country])):
        print(f"预测值: {predictions[country][i]:.2f}, 实际值: {actual[country][i]:.2f}")
        mse = np.mean(np.square(np.array(predictions[country]), np.array(actual[country])))
        print(f"预测值和实际值方差：{mse:.2f}")

# 可视化预测结果
plt.figure(figsize=(14, 6))
for i, country in enumerate(sheet_names, 1):
    plt.subplot(1, 2, i)
    plt.plot(actual[country], label='Actual', marker='o')
    plt.plot(predictions[country], label='Prediction', marker='x')
    plt.xlabel('Samples')
    plt.ylabel('GDP (Trillion USD)')
    plt.title(f'{country} GDP Prediction: Actual vs Predicted')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()

# 预测
# 获取每个国家最新的 sequence_length 个 GDP_scaled 数据
last_sequences = {}
for country in sheet_names:
    country_seq = country_data[country_data['Country'] == country]['GDP_scaled'].values
    last_seq = country_seq[-sequence_length:]
    last_sequences[country] = list(last_seq)
    print(f"{country} - 最后一个序列: {last_seq}")


# 定义预测函数（与上文相同）
def predict_future_gdp(model, last_seq, scaler, device, sequence_length=5, future_steps=5):
    model.eval()
    future_gdp = []
    current_seq = last_seq.copy()
    for _ in range(future_steps):
        input_seq = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            y_pred = model(input_seq)
            y_pred = y_pred.cpu().numpy()
            y_pred_original = scaler.inverse_transform(y_pred)
            predicted_gdp = y_pred_original.flatten()[0]
            future_gdp.append(predicted_gdp)
        y_pred_scaled = scaler.transform([[predicted_gdp]])[0][0]
        current_seq.append(y_pred_scaled)
        current_seq.pop(0)
    return future_gdp


#  执行预测
last_years = country_data.groupby('Country')['Year'].max().to_dict()
future_years = {country: list(range(last_years[country] + 1, last_years[country] + 6)) for country in sheet_names}
future_predictions = {country: [] for country in sheet_names}

for country in sheet_names:
    model = models[country]
    scaler_country = MinMaxScaler(feature_range=(0, 1))
    country_subset = country_data[country_data['Country'] == country]
    scaler_country.fit(country_subset[['GDP']])
    future_gdp = predict_future_gdp(
        model=model,
        last_seq=last_sequences[country],
        scaler=scaler_country,
        device=device,
        sequence_length=sequence_length,
        future_steps=5
    )
    future_predictions[country] = future_gdp
    print(f"{country} - 未来5年预测GDP: {future_gdp}")

# 可视化预测结果
future_data_frames = []
for country in sheet_names:
    df_future = pd.DataFrame({
        'Country': country,
        'Year': future_years[country],
        'GDP': future_predictions[country]
    })
    future_data_frames.append(df_future)

future_data = pd.concat(future_data_frames, ignore_index=True)
print(future_data)

plt.figure(figsize=(12, 6))
for country in sheet_names:
    # 历史数据
    country_subset = country_data[country_data['Country'] == country]
    plt.plot(country_subset['Year'], country_subset['GDP'],
             marker='o', label=f'{country} Actual')

    # 预测数据
    country_future = future_data[future_data['Country'] == country]
    plt.plot(country_future['Year'], country_future['GDP'],
             marker='x', linestyle='--', label=f'{country} Predicted')
plt.title('GDP Over Years for China and United States (Actual vs Predicted)')
plt.xlabel('Year')
plt.ylabel('GDP (Trillion USD)')
plt.legend()
plt.grid()
plt.show()
