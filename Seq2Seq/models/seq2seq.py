# models/seq2seq.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils import PAD_token

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]

        # Pack padded batch of sequences for RNN module
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded.permute(1, 0, 2), src_lengths, enforce_sorted=False)

        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)  # [src_len, batch_size, hid_dim * 2]

        # Concatenate the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hid_dim * 2]

        # Pass through a linear layer
        hidden = torch.tanh(self.fc(hidden))  # [batch_size, hid_dim]

        return outputs, hidden.unsqueeze(0)  # outputs: [src_len, batch_size, hid_dim * 2], hidden: [1, batch_size, hid_dim]

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        src_len = encoder_outputs.shape[0]

        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2]

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]

        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)  # [batch_size, src_len]

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        input = input.unsqueeze(0)  # [1, batch_size]

        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        # Calculate attention weights
        a = self.attention(hidden[-1], encoder_outputs, mask)  # [batch_size, src_len]

        a = a.unsqueeze(1)  # [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2]

        # Weighted sum of encoder outputs
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim * 2]

        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, hid_dim * 2]

        # Concatenate embedded input and weighted encoder outputs
        lstm_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, (hid_dim * 2) + emb_dim]

        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, torch.zeros_like(hidden)))

        # Concatenate output, weighted encoder outputs, and embedded input
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        output = output.squeeze(0)      # [batch_size, hid_dim]
        weighted = weighted.squeeze(0)  # [batch_size, hid_dim * 2]

        fc_input = torch.cat((output, weighted, embedded), dim=1)  # [batch_size, (hid_dim * 2) + hid_dim + emb_dim]

        prediction = self.fc_out(fc_input)  # [batch_size, output_dim]

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        # src: [batch_size, src_len]
        mask = (src != self.src_pad_idx).permute(1, 0)  # [src_len, batch_size]
        return mask

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # src_lengths: [batch_size]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # Initial input to decoder is SOS tokens
        input = trg[:,0]  # [batch_size]

        mask = self.create_mask(src)

        for t in range(1, trg_len):
            # Decode
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)

            # Store output
            outputs[t] = output

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token
            top1 = output.argmax(1)

            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = trg[:,t] if teacher_force else top1

        return outputs
