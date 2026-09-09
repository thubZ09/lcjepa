from __future__ import annotations
import numpy as np
import torch
from joblib import Parallel, delayed
from torch import Tensor
from torch.utils.data import Dataset
from core.config import DataConfig
from core.data.drw import sample_params, simulate_drw

def _simulate_one(seed: int, cfg: DataConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_tau, log_sigma = sample_params(cfg.log_tau_range, cfg.log_sigma_range, rng)
    lc = simulate_drw(
        seq_len=cfg.seq_len,
        log_tau=log_tau,
        log_sigma=log_sigma,
        cadence_days=cfg.cadence_days,
        cadence_jitter=cfg.cadence_jitter,
        season_gap_frac=cfg.season_gap_frac,
        phot_err=cfg.phot_err,
        rng=rng,
    )
    return {
        "t": lc.t,
        "mag": lc.mag,
        "mag_err": lc.mag_err,
        "labels": np.array([lc.log_tau, lc.log_sigma], dtype=np.float32),
    }

class SyntheticDRWDataset(Dataset):
    def __init__(self, cfg: DataConfig, n: int, base_seed: int):
        self.cfg = cfg
        seeds = [base_seed + i for i in range(n)]
        records = Parallel(n_jobs=cfg.n_jobs, prefer="processes")(
            delayed(_simulate_one)(s, cfg) for s in seeds
        )
        self._records: list[dict[str, np.ndarray]] = records  
    def __len__(self) -> int:
        return len(self._records)
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        r = self._records[idx]
        t = r["t"]
        mag = r["mag"]
        dt = (t - t[0]).astype(np.float32)
        mag_mean = float(mag.mean())
        mag_std = float(mag.std() + 1e-6)
        mag_norm = (mag - mag_mean) / mag_std
        return {
            "dt": torch.from_numpy(dt),                  
            "mag": torch.from_numpy(mag_norm.astype(np.float32)),  
            "mag_err": torch.from_numpy(r["mag_err"] / mag_std),  
            "labels": torch.from_numpy(r["labels"]),      
        }

def collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "dt": torch.stack([b["dt"] for b in batch]),
        "mag": torch.stack([b["mag"] for b in batch]),
        "mag_err": torch.stack([b["mag_err"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }