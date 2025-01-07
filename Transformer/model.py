# -*- coding: utf-8 -*-
"""
model.py
手撕Transformer模型定义
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class PositionalEncoding(nn.Module):
    """
    位置编码，帮助模型利用序列的位置信息
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算 [max_len, d_model] 的位置编码矩阵 pe
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        # 计算 1/10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 偶数维用 sin, 奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 将 pe 注册为 buffer，表示不参与训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 将位置编码加到输入 x 上
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力
    """

    def __init__(self, dim_model, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = dim_model // num_heads

        self.query = nn.Linear(dim_model, dim_model)
        self.key = nn.Linear(dim_model, dim_model)
        self.value = nn.Linear(dim_model, dim_model)

        self.fc = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        :param x: [batch_size, seq_len, dim_model]
        :param mask: [batch_size, seq_len], 这里简化为不使用mask 或自行构造
        :return: [batch_size, seq_len, dim_model]
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性变换 Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, dim_model]
        K = self.key(x)
        V = self.value(x)

        # 2. 拆分为 num_heads 个头，并把 head 维度合并到 batch_size
        #    [batch_size, seq_len, num_heads, head_dim] -> [batch_size * num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数 score = Q*K^T / sqrt(d_k)
        #    Q, K 维度：[batch_size * num_heads, seq_len, head_dim]
        #    score:   [batch_size * num_heads, seq_len, seq_len]
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 可选：若有 mask，这里对 score 做 -inf 填充或相应操作
        if mask is not None:
            # mask = [batch_size, seq_len], 需要扩展到 [batch_size * num_heads, seq_len, seq_len]
            # 不过在情感分析任务中，一般可能不需要 mask，或仅做 padding mask
            # 这里只作示例
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # [batch_size, num_heads, seq_len]
            mask = mask.unsqueeze(-2)  # [batch_size, num_heads, seq_len, 1]
            mask = mask.repeat(1, 1, 1, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
            mask = mask.view(batch_size * self.num_heads, seq_len, seq_len)
            score = score.masked_fill(mask == 0, -1e9)

        # 5. 对 score 做 softmax
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # 6. 加权求和得到上下文向量
        context = torch.matmul(attn, V)  # [batch_size*num_heads, seq_len, head_dim]

        # 7. 重塑形状 [batch_size, seq_len, dim_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_model)

        # 8. 通过最后的线性层
        out = self.fc(context)
        return out


class FeedForward(nn.Module):
    """
    前馈网络
    """

    def __init__(self, dim_model, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Encoder中包含 Multi-head Attention + Add & Norm + FeedForward + Add & Norm
    """

    def __init__(self, dim_model, num_heads, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dim_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)

        self.ffn = FeedForward(dim_model, hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        # Multi-head Attention
        mha_out = self.mha(x, mask)
        x = x + self.dropout1(mha_out)
        x = self.norm1(x)

        # Feed Forward
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """
    堆叠多个EncoderLayer
    """

    def __init__(self, dim_model, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerClassifier(nn.Module):
    """
    手撕Transformer的文本分类模型
    1. 词嵌入(Embedding)
    2. 位置编码(PositionalEncoding)
    3. 若干层Encoder
    4. 全连接分类
    """

    def __init__(self, config: Config):
        super(TransformerClassifier, self).__init__()
        self.config = config

        # 词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed,
            padding_idx=0  # <PAD>默认为0
        )

        # 位置编码
        self.position_encoding = PositionalEncoding(
            d_model=config.dim_model,
            max_len=config.pad_size,
            dropout=config.dropout
        )

        # 转换 Embedding 输出维度 -> dim_model(若 embed != dim_model 可加线性映射)
        # 为简化，这里假设 embed == dim_model，否则需加一层 Linear
        if config.embed != config.dim_model:
            self.embedding_transform = nn.Linear(config.embed, config.dim_model)
        else:
            self.embedding_transform = None

        # 编码器
        self.encoder = TransformerEncoder(
            dim_model=config.dim_model,
            num_heads=config.num_head,
            hidden_dim=config.hidden,
            num_layers=config.num_encoder,
            dropout=config.dropout
        )

        # 分类器：将 [batch_size, seq_len, dim_model] -> [batch_size, num_classes]
        # 做法1：先 reshape -> [batch_size, seq_len * dim_model]
        # 做法2：池化(如 mean pooling / max pooling)
        # 这里与示例一致，用 view 展开后全连接
        self.fc = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len], 这里是索引序列
        :return: logits: [batch_size, num_classes]
        """
        # 1. 词嵌入
        out = self.embedding(x)  # [batch_size, seq_len, embed]

        # 如果需要将embed -> dim_model
        if self.embedding_transform is not None:
            out = self.embedding_transform(out)  # [batch_size, seq_len, dim_model]

        # 2. 加位置编码
        out = self.position_encoding(out)  # [batch_size, seq_len, dim_model]

        # 3. 多层Encoder
        out = self.encoder(out)  # [batch_size, seq_len, dim_model]

        # 4. 将三维张量展平 [batch_size, seq_len * dim_model]
        out = out.view(out.size(0), -1)

        # 5. 全连接分类
        out = self.fc(out)  # [batch_size, num_classes]
        return out
