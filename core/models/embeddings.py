from __future__ import annotations
import math
import torch
from torch import Tensor, nn

class FourierTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, n_fourier: int = 32, max_period: float = 4096.0):
        super().__init__()
        freqs = torch.exp(
            torch.linspace(math.log(2 * math.pi / max_period), math.log(2 * math.pi), n_fourier)
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * n_fourier, d_model)
    def forward(self, dt: Tensor) -> Tensor:  
        ang = dt.unsqueeze(-1) * self.freqs  
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)  
        return self.proj(feats)

class Time2Vec(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, dt: Tensor) -> Tensor:  
        x = dt.unsqueeze(-1) * self.w + self.b
        out = x.clone()
        out[..., 1:] = torch.sin(x[..., 1:]) 
        return out

class ObservationEmbedding(nn.Module):
    def __init__(self, d_model: int, time_embed: str = "fourier", n_fourier: int = 32):
        super().__init__()
        if time_embed == "fourier":
            self.time = FourierTimeEmbedding(d_model, n_fourier)
        elif time_embed == "time2vec":
            self.time = Time2Vec(d_model)
        else:
            raise ValueError(f"unknown time_embed: {time_embed}")
        self.value = nn.Linear(2, d_model)  
        self.norm = nn.LayerNorm(d_model)
    def forward(self, dt: Tensor, mag: Tensor, mag_err: Tensor) -> Tensor:
        v = self.value(torch.stack([mag, mag_err], dim=-1)) 
        return self.norm(self.time(dt) + v)