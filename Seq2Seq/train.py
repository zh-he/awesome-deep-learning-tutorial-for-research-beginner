import torch
from tqdm import tqdm


def train_model(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for i, (src, trg, src_lengths, trg_lengths) in enumerate(tqdm(iterator, desc="Training")):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        # forward
        output = model(src, src_lengths, trg, teacher_forcing_ratio=0.5)
        # output: [trg_len, batch_size, output_dim]
        # trg:    [batch_size, trg_len]

        output_dim = output.shape[-1]
        # 去掉第一个 time step (它只是根据 SOS 输出)，并摊平
        output = output[1:].view(-1, output_dim)  # [(trg_len -1) * batch_size, output_dim]
        trg = trg[:, 1:].reshape(-1)  # [(trg_len -1) * batch_size]

        loss = criterion(output, trg)

        # 计算准确率
        preds = output.argmax(1)
        correct = (preds == trg).float()
        acc = correct.sum() / len(correct)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_model(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for i, (src, trg, src_lengths, trg_lengths) in enumerate(tqdm(iterator, desc="Evaluating")):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            preds = output.argmax(1)
            correct = (preds == trg).float()
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_seq2seq(model, train_loader, valid_loader, optimizer, criterion, N_EPOCHS, CLIP, device):
    import time
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, CLIP, device)
        valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 保存最优模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tVal.  Loss: {valid_loss:.3f} | Val.  Acc: {valid_acc * 100:.2f}%')
