from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from core.config import DataConfig

def _crop_or_window(n: int, seq_len: int, rng: np.random.Generator, train: bool) -> slice:
    if n <= seq_len:
        return slice(0, n) 
    start = int(rng.integers(0, n - seq_len + 1)) if train else (n - seq_len) // 2
    return slice(start, start + seq_len)

class ZTFLightCurveDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str, base_seed: int = 0):
        try:
            import pandas as pd
        except ImportError as e: 
            raise ImportError(
                "support needs the 'data' extra: `uv pip install -e '.[data]'`"
            ) from e
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.train = split == "train"   
        self._rng = np.random.default_rng(base_seed)
        cache = Path(cfg.ztf_cache_dir)
        files = sorted(cache.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet in {cache}. Run scripts/download_ztf.py first."
            )
        min_epochs = max(cfg.ztf_min_epochs, cfg.seq_len)
        records: list[dict] = []
        oids: list[int] = []
        n_too_short = 0
        for f in files:
            df = pd.read_parquet(f)
            if len(df) < min_epochs:
                n_too_short += 1
                continue
            records.append(
                {
                    "mjd": df["mjd"].to_numpy(np.float64),
                    "mag": df["mag"].to_numpy(np.float32),
                    "magerr": df["magerr"].to_numpy(np.float32),
                    "label": float(df["label"].iloc[0]) if "label" in df.columns else np.nan,
                }
            )
            try:
                oids.append(int(f.stem))
            except ValueError:
                oids.append(-1)
        idx = np.arange(len(records))
        self._rng.shuffle(idx)
        n_val = int(round(cfg.ztf_val_frac * len(records)))
        val_idx = set(idx[:n_val].tolist())
        if split == "all":
            keep = list(range(len(records)))
        else:
            keep = [i for i in idx if (i in val_idx) == (split == "val")]
        self._records = [records[i] for i in keep]
        self.oids = [oids[i] for i in keep]
        if not self._records:
            raise RuntimeError(
                f"No curves with >= {min_epochs} epochs for split={split} "
                f"({n_too_short} of {len(files)} files were too short). "
                f"Lower data.seq_len or download targets with longer baselines."
            )
        if split in ("train", "all") and n_too_short:
            print(f"[ztf] dropped {n_too_short}/{len(files)} curves with < {min_epochs} epochs")

    def __len__(self) -> int:
        return len(self._records)
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        r = self._records[idx]
        n = r["mjd"].size
        sl = _crop_or_window(n, self.seq_len, self._rng, self.train)
        mjd = r["mjd"][sl]
        mag = r["mag"][sl]
        magerr = r["magerr"][sl]
        if mjd.size < self.seq_len:
            pad = self.seq_len - mjd.size
            mjd = np.concatenate([mjd, np.full(pad, mjd[-1])])
            mag = np.concatenate([mag, np.full(pad, mag[-1])])
            magerr = np.concatenate([magerr, np.full(pad, magerr[-1])])
        dt = (mjd - mjd[0]).astype(np.float32)
        mag_mean = float(mag.mean())
        mag_std = float(mag.std() + 1e-6)
        mag_norm = (mag - mag_mean) / mag_std
        labels = np.array([r["label"], np.nan], dtype=np.float32)
        return {
            "dt": torch.from_numpy(dt),
            "mag": torch.from_numpy(mag_norm.astype(np.float32)),
            "mag_err": torch.from_numpy((magerr / mag_std).astype(np.float32)),
            "labels": torch.from_numpy(labels),
        }