import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 用于显示进度条


def init_weights(m):
    """
    初始化模型参数
    """
    for name, param in m.named_parameters():
        if param.requires_grad:
            nn.init.uniform_(param.data, -0.08, 0.08)


def train_one_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(dataloader, desc="Training", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)  # [trg_len, batch_size, output_dim]
        # output 与 trg 的 shape 对齐
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)  # 去掉第0个时刻 <bos>
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 防止梯度爆炸
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc="Evaluating", leave=False):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def train_model(model, train_dataloader, val_dataloader, device,
                n_epochs=20, clip=1, lr=1e-3, save_path='best_model.pt'):
    """
    整体训练流程，包括保存最佳模型
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 这里默认为 <pad> 的索引为0，需根据实际情况修改

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}:")
        # 训练
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, clip, device)
        # 验证
        val_loss = evaluate(model, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  >> Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

        # 如果验证集更好，则保存当前权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("  >> Best model saved.")

    return train_losses, val_losses
