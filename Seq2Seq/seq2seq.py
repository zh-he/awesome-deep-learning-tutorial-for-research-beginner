import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    编码器
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch_size, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell:   [n_layers, batch_size, hid_dim]
        return outputs, hidden, cell


class Attention(nn.Module):
    """
    简单的 Luong Attention 实现
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [n_layers, batch_size, hid_dim], 这里取最后一层 hidden 用于注意力计算
        encoder_outputs: [src_len, batch_size, hid_dim]
        返回： [batch_size, src_len] 的注意力权重分布
        """
        # 取 Decoder 最上层隐藏状态: hidden[-1] => [batch_size, hid_dim]
        hidden = hidden[-1]
        src_len = encoder_outputs.shape[0]

        # encoder_outputs: [batch_size, src_len, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden: [batch_size, hid_dim] => [batch_size, 1, hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 拼接后通过一个线性层
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]

        # 之后再过一个线性层 v => [batch_size, src_len, 1] => squeeze => [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    解码器，加入 Attention
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        input: [batch_size]
        hidden, cell: [n_layers, batch_size, hid_dim]
        encoder_outputs: [src_len, batch_size, hid_dim]
        """
        input = input.unsqueeze(0)  # => [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # => [1, batch_size, emb_dim]

        # 计算注意力权重
        a = self.attention(hidden, encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # => [batch_size, 1, src_len]

        # encoder_outputs: [batch_size, src_len, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # 加权求和得到 context
        context = torch.bmm(a, encoder_outputs)  # => [batch_size, 1, hid_dim]
        context = context.permute(1, 0, 2)  # => [1, batch_size, hid_dim]

        # rnn 的输入是 [emb_dim + hid_dim]
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # output: [1, batch_size, hid_dim]
        # 拼接 output 和 context
        output = torch.cat((output.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(output)  # => [batch_size, output_dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    将 Encoder 和 Decoder 组合到一起
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: [src_len, batch_size]
        trg: [trg_len, batch_size]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # 编码器
        encoder_outputs, hidden, cell = self.encoder(src)

        # 准备一个容器来存放解码器的输出
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 解码器第一个输入是 <bos>
        input = trg[0, :]

        for t in range(1, trg_len):
            # 将 encoder_outputs 也传给解码器，用于注意力
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output

            # 选出概率最大的词
            top1 = output.argmax(1)
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1

        return outputs
