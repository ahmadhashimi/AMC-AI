"""Readable, educational GPT-style architecture; no pretrained weights.

Explicit Q/K/V matrices, masked causal self-attention, MLPs, layer norms,
residual connections and tied token embeddings are present for inspection.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 257
    context_length: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_model: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        batch, seq, d_model = x.shape
        # Split the learned projection into query, key and value matrices.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask: a token cannot read future tokens.
        future = torch.ones(seq, seq, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        return self.dropout(self.out(out))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AMCTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embeddings = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight  # tied weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        batch, seq = input_ids.shape
        if seq > self.cfg.context_length:
            raise ValueError(f"sequence {seq} exceeds context length {self.cfg.context_length}")
        positions = torch.arange(seq, device=input_ids.device)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
