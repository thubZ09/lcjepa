from __future__ import annotations
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class DataConfig:
    source: str = "synthetic"  
    seq_len: int = 200          
    n_train: int = 8_000
    n_val: int = 1_000
    log_tau_range: tuple[float, float] = (1.5, 3.0)   
    log_sigma_range: tuple[float, float] = (-1.3, -0.3)  
    cadence_days: float = 3.0    
    cadence_jitter: float = 1.0 
    season_gap_frac: float = 0.25 
    phot_err: float = 0.02       
    n_jobs: int = -1            
    ztf_cache_dir: str = "data/ztf_cache"  
    ztf_band: str = "r"         
    ztf_min_epochs: int = 120   
    ztf_val_frac: float = 0.1     

@dataclass
class MaskConfig:
    n_target_blocks: int = 4
    target_block_frac: tuple[float, float] = (0.10, 0.20)  
    context_keep_frac: float = 0.85  

@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    enc_depth: int = 4
    pred_depth: int = 2
    rollout_depth: int = 2     
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    time_embed: str = "fourier"   
    n_fourier: int = 32           

@dataclass
class LossConfig:
    objective: str = "jepa"      
    pred_weight: float = 1.0      
    rollout_weight: float = 0.1   
    vic_var_weight: float = 1.0 
    vic_cov_weight: float = 0.04 
    vic_var_target: float = 1.0  

@dataclass
class OptimConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1.5e-3
    weight_decay: float = 0.04
    warmup_epochs: int = 3
    ema_base: float = 0.996      
    ema_final: float = 1.0   
    grad_clip: float = 1.0

@dataclass
class RunConfig:
    seed: int = 0
    device: str | None = None   
    out_dir: str = "runs/synthetic"
    log_every: int = 20
    probe_every: int = 5    

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
    @staticmethod
    def from_yaml(path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return _build(Config, raw)
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

def _build(cls: type, raw: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        ftype = hints.get(f.name, f.type)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        elif isinstance(value, list):
            kwargs[f.name] = tuple(value) 
        else:
            kwargs[f.name] = value
    return cls(**kwargs)