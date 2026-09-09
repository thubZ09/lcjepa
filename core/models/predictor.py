from __future__ import annotations
import torch
from torch import Tensor, nn
from core.models.encoder import TransformerEncoder

class JEPAPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.transformer = TransformerEncoder(d_model, n_heads, depth, mlp_ratio, dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, context: Tensor, target_time_embed: Tensor) -> Tensor:
        b, lt, _ = target_time_embed.shape
        queries = self.mask_token.expand(b, lt, -1) + target_time_embed
        x = torch.cat([context, queries], dim=1)
        x = self.transformer(x)
        pred = x[:, -lt:, :] 
        return self.out(pred)

class LatentRollout(nn.Module):
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
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward=int(d_model * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, z: Tensor) -> Tensor:
        mask = self._causal_mask(z.size(1), z.device)
        h = z
        for blk in self.blocks:
            h = blk(h, src_mask=mask)
        return self.out(self.norm(h))