import math
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Token embedding + sinusoidal positional encoding."""

    def __init__(self, vocabulary_size, embedding_dim, max_len=5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)

        pe = torch.zeros(max_len, embedding_dim)
        positions = torch.arange(0, max_len).unsqueeze(1).float()

        div = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        emb = self.embedding_layer(x)
        emb = emb + self.pe[:, : x.size(1), :]
        return emb


class MultiHeadSelfAttentionBlock(nn.Module):
    """Multi-head self attention with optional causal masking."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d = embedding_dim
        self.h = num_heads
        assert self.d % self.h == 0, "Embedding dim must be divisible by num_heads"
        self.dh = self.d // self.h

        self.Q = nn.Linear(self.d, self.d)
        self.K = nn.Linear(self.d, self.d)
        self.V = nn.Linear(self.d, self.d)
        self.dropout = nn.Dropout(dropout)
        self.output_lin = nn.Linear(self.d, self.d)

    def compute_attention_scores(self, Q: torch.Tensor, K: torch.Tensor):
        return Q @ K.transpose(-2, -1) / math.sqrt(self.dh)

    def forward(self, x_embed: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, _ = x_embed.shape

        q = self.Q(x_embed)
        k = self.K(x_embed)
        v = self.V(x_embed)

        q = q.view(batch_size, seq_len, self.h, self.dh).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.h, self.dh).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.h, self.dh).transpose(1, 2)

        scores = self.compute_attention_scores(q, k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d)

        return self.output_lin(output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with 4x expansion."""

    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear_2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional causal masking."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        masked: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.masked = masked
        self.multihead_attn = MultiHeadSelfAttentionBlock(
            embedding_dim, num_heads, dropout
        )
        self.norm_layer_1 = nn.LayerNorm(embedding_dim)
        self.norm_layer_2 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, dropout)

    def forward(self, x: torch.Tensor):
        mask = None
        if self.masked:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        attn = self.multihead_attn(self.norm_layer_1(x), mask=mask)
        x = x + attn
        ff = self.feedforward(self.norm_layer_2(x))
        x = x + ff
        return x
