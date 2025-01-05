import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD_token


class Encoder(nn.Module):
    """
    简化版：单向 LSTM，不做额外的线性映射。
    直接将最后一层的 hidden & cell 作为输出，便于 Decoder 初始化。
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
                   # 仅在除最后一层外的层之间生效
            bidirectional=False     # 关键：单向
        )


    def forward(self, src, src_lengths):
        """
        src: [batch_size, src_len]
        src_lengths: [batch_size], 记录每个句子的实际长度（可用于 pack/pad）
        """
        # [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        # pack_padded_sequence 需 [seq_len, batch_size, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded.permute(1, 0, 2),  # (src_len, batch_size, emb_dim)
            src_lengths,
            enforce_sorted=False
        )

        # outputs: [src_len, batch_size, hid_dim] (因为单向)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs: [src_len, batch_size, hid_dim]

        # hidden, cell: [n_layers, batch_size, hid_dim]
        # 如果 n_layers=1，则 hidden[-1]、cell[-1] 即可表示最后一步状态

        return outputs, hidden, cell


class Attention(nn.Module):
    """
    与之前相同的加性 Attention (类似 Bahdanau)，
    仍然可保留，以提升翻译质量。
    """
    def __init__(self, hid_dim):
        super().__init__()
        # 对 (decoder_hidden + encoder_output) 做线性映射
        self.attn = nn.Linear(hid_dim + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_hidden, encoder_outputs, mask):
        """
        dec_hidden: [batch_size, hid_dim], Decoder当前最后一层的隐藏状态
        encoder_outputs: [src_len, batch_size, hid_dim], Encoder输出序列
        mask: [batch_size, src_len], 1 表示有效位置，0 表示 PAD
        """
        src_len = encoder_outputs.shape[0]

        # 让 dec_hidden 与每个 time step 的 encoder_outputs 拼接
        # dec_hidden => [batch_size, 1, hid_dim] => repeat => [batch_size, src_len, hid_dim]
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 维度对齐: encoder_outputs => [batch_size, src_len, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy: [batch_size, src_len, hid_dim]
        energy = torch.tanh(
            self.attn(torch.cat((dec_hidden, encoder_outputs), dim=2))
        )

        # attention: [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        # 对PAD位置mask掉
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    简化版 Decoder: 单向 LSTM + Attention (可选)。
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, attention=None):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention  # 可以为 None，则不使用注意力

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_token)
        # Decoder LSTM 输入: (attended context + emb_dim)
        input_dim_for_lstm = emb_dim if attention is None else (emb_dim + hid_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim_for_lstm,
            hidden_size=hid_dim,
            num_layers=n_layers,
            bidirectional=False
        )
        # 输出层: 结合了 LSTM输出 + 上一步的 context + embedding
        # 如果不使用 Attention，则少一个 hid_dim
        fc_in_dim = hid_dim + emb_dim
        if attention is not None:
            fc_in_dim += hid_dim

        self.fc_out = nn.Linear(fc_in_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        """
        input: [batch_size], 解码器当前时刻输入的 token
        hidden, cell: [n_layers, batch_size, hid_dim]
        encoder_outputs: [src_len, batch_size, hid_dim]
        mask: [batch_size, src_len]
        """
        # [1, batch_size]
        input = input.unsqueeze(0)

        # [1, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(input))

        # 如果有注意力:
        if self.attention is not None:
            # dec_hidden 用 hidden[-1], shape: [batch_size, hid_dim]
            dec_hidden = hidden[-1]
            # [batch_size, src_len]
            a = self.attention(dec_hidden, encoder_outputs, mask)
            a = a.unsqueeze(1)  # [batch_size, 1, src_len]

            # encoder_outputs => [batch_size, src_len, hid_dim]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # [batch_size, 1, hid_dim]
            context = torch.bmm(a, encoder_outputs)
            # => [1, batch_size, hid_dim]
            context = context.permute(1, 0, 2)

            # 作为 LSTM 输入: [1, batch_size, emb_dim + hid_dim]
            rnn_input = torch.cat((embedded, context), dim=2)
        else:
            # 不使用注意力
            rnn_input = embedded

        # 经过 LSTM
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # output: [1, batch_size, hid_dim]
        # hidden, cell: [n_layers, batch_size, hid_dim]

        # 计算输出
        output = output.squeeze(0)  # [batch_size, hid_dim]
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]

        if self.attention is not None:
            context = context.squeeze(0)  # [batch_size, hid_dim]
            fc_input = torch.cat((output, context, embedded), dim=1)  # [batch_size, hid_dim+hid_dim+emb_dim]
        else:
            fc_input = torch.cat((output, embedded), dim=1)           # [batch_size, hid_dim+emb_dim]

        # [batch_size, output_dim]
        prediction = self.fc_out(fc_input)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        # src: [batch_size, src_len]
        return (src != self.src_pad_idx)  # [batch_size, src_len]

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        src: [batch_size, src_len]
        src_lengths: [batch_size]
        trg: [batch_size, trg_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # decoder outputs: [trg_len, batch_size, trg_vocab_size]
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 1) 编码
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        # hidden, cell => [n_layers, batch_size, hid_dim]

        # 2) Decoder初始输入: trg[:, 0] (通常是SOS)
        input = trg[:, 0]
        mask = self.create_mask(src)  # [batch_size, src_len]

        # 3) 开始循环解码
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[t] = output
            # 计算 teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # [batch_size]

            # 下一个输入
            input = trg[:, t] if teacher_force else top1

        return outputs