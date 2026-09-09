from __future__ import annotations
import torch
from torch import Tensor, nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)