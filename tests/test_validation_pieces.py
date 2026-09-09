from __future__ import annotations
import numpy as np
from core.data.drw import simulate_clagn, simulate_drw
from core.anomaly import Features, _shape_features, supervised_score

def test_clagn_has_larger_step_than_normal():
    diffs_normal, diffs_clagn = [], []
    for s in range(40):
        rng = np.random.default_rng(s)
        lc = simulate_drw(200, 2.2, -0.5, 3.0, 1.0, 0.25, 0.02, rng)
        h = lc.mag.size // 2
        diffs_normal.append(abs(lc.mag[:h].mean() - lc.mag[h:].mean()))
        rng = np.random.default_rng(s)
        cl = simulate_clagn(200, 2.2, -0.5, 3.0, 1.0, 0.25, 0.02, rng)
        h = cl.mag.size // 2
        diffs_clagn.append(abs(cl.mag[:h].mean() - cl.mag[h:].mean()))
    assert np.median(diffs_clagn) > np.median(diffs_normal)

def test_shape_features_detect_step():
    rng = np.random.default_rng(0)
    flat = rng.normal(0, 1, 200).astype(np.float32)
    step = flat.copy()
    step[100:] += 5.0 
    assert _shape_features(step)[0] > _shape_features(flat)[0]
    assert _shape_features(step)[1] > _shape_features(flat)[1]

def test_supervised_score_is_oof_and_finite():
    rng = np.random.default_rng(0)
    n = 120
    labels = np.array([0, 1] * (n // 2), dtype=float)
    emb = rng.normal(size=(n, 8))
    surprise = rng.normal(size=n)
    shape = np.stack([labels + rng.normal(0, 0.3, n), rng.normal(0, 1, n)], axis=1)
    feats = Features(embeddings=emb, surprise=surprise, labels=labels, shape=shape)
    scores = supervised_score(feats, seed=0)
    assert np.all(np.isfinite(scores))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(labels, scores) > 0.7  

def test_curves_from_dataframe_adapter():
    import pandas as pd
    df = pd.DataFrame(
        {
            "oid": [1, 1, 1, 2],
            "mjd": [3.0, 1.0, 2.0, 5.0],
            "mag": [18.0, 18.1, 17.9, 19.0],
            "magerr": [0.03, 0.5, 0.04, 0.02], 
            "filtercode": ["zr", "zr", "zr", "zr"],
            "ra": [10.0, 10.0, 10.0, 10.0],
            "dec": [5.0, 5.0, 5.0, 5.0],
        }
    )
    from core.data.ztf import curves_from_dataframe
    curves = curves_from_dataframe(df, band="r", max_magerr=0.2, label="1")
    by_oid = {c.oid: c for c in curves}
    assert by_oid[1].mjd.size == 2
    assert np.all(np.diff(by_oid[1].mjd) > 0)