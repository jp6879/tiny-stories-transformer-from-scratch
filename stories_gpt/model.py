import torch
from torch import nn
from .components import PositionalEmbedding, TransformerBlock


class GPT(nn.Module):
    """Decoder-only transformer for next-token prediction."""

    def __init__(
        self, vocab_size, embedding_dim, num_heads, num_layers, max_len, dropout=0.1
    ):
        super().__init__()
        self.max_len = max_len

        self.embedding = PositionalEmbedding(vocab_size, embedding_dim, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, masked=True, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # only feed the last max_len tokens if the sequence grew too long
            ctx = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]

            logits = self(ctx)[:, -1, :] / temperature

            if top_k is not None:
                cutoff, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < cutoff[:, [-1]]] = float("-inf")

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
