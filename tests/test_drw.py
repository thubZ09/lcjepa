from __future__ import annotations
import numpy as np
from core.config import DataConfig
from core.data.drw import simulate_drw
from core.data.synthetic import SyntheticDRWDataset, collate

def test_drw_recovers_variance_scale():
    rng = np.random.default_rng(0)
    lc_short = simulate_drw(2000, 1.5, -0.5, 3.0, 0.5, 0.0, 0.0, rng)
    rng = np.random.default_rng(0)
    lc_long = simulate_drw(2000, 3.0, -0.5, 3.0, 0.5, 0.0, 0.0, rng)
    assert np.var(np.diff(lc_long.mag)) < np.var(np.diff(lc_short.mag))

def test_times_monotonic():
    rng = np.random.default_rng(1)
    lc = simulate_drw(500, 2.0, -0.5, 3.0, 1.0, 0.25, 0.02, rng)
    assert np.all(np.diff(lc.t) > 0)

def test_dataset_batch_shapes():
    cfg = DataConfig(seq_len=64, n_train=8, n_jobs=1)
    ds = SyntheticDRWDataset(cfg, n=8, base_seed=0)
    batch = collate([ds[i] for i in range(4)])
    assert batch["mag"].shape == (4, 64)
    assert batch["labels"].shape == (4, 2)