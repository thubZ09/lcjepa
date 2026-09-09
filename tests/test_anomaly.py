from __future__ import annotations
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from core.config import Config
from core.data.synthetic import collate
from core.anomaly import combined_score, evaluate, extract_features, ranked_candidates
from core.models.lcjepa import LCJEPA

def _make_cache(tmp_path, n_clagn=6, n_ctrl=6):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(0)
    oid = 0
    for label, amp in [(1, 0.4), (0, 0.08)]:  
        n_obj = n_clagn if label == 1 else n_ctrl
        for _ in range(n_obj):
            n = 200
            df = pd.DataFrame(
                {
                    "mjd": np.cumsum(rng.uniform(1, 3, n)) + 58000,
                    "mag": (18 + rng.normal(0, amp, n)).astype(np.float32),
                    "magerr": np.full(n, 0.03, np.float32),
                    "label": np.full(n, float(label)),
                }
            )
            df.to_parquet(tmp_path / f"{oid}.parquet", index=False)
            oid += 1

def test_anomaly_pipeline_runs(tmp_path):
    _make_cache(tmp_path)
    from core.data.ztf_dataset import ZTFLightCurveDataset
    cfg = Config()
    cfg.data.source = "ztf"
    cfg.data.seq_len = 128
    cfg.data.ztf_cache_dir = str(tmp_path)
    cfg.data.ztf_min_epochs = 120
    cfg.model.d_model = 32
    cfg.model.enc_depth = 2
    ds = ZTFLightCurveDataset(cfg.data, split="all", base_seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    model = LCJEPA(cfg)
    feats = extract_features(model, loader, torch.device("cpu"))
    assert feats.embeddings.shape[0] == len(ds)
    assert feats.surprise.shape[0] == len(ds)
    scores = combined_score(feats)
    assert scores.shape[0] == len(ds)
    metrics = evaluate(scores, feats.labels)
    assert "roc_auc" in metrics  
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    rows = ranked_candidates(ds.oids, scores, feats.labels, top=5)
    assert len(rows) == 5
    assert rows[0]["rank"] == 1