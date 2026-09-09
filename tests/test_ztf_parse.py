from __future__ import annotations
import numpy as np
import pytest
from core.config import DataConfig
from core.data.ztf import best_curve, build_query_url, parse_lightcurve_csv

MOCK_CSV = """oid,expid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec
100,1,2458000.5,58000.0,18.20,0.03,0,zr,150.0,2.0
100,2,2458010.5,58010.0,18.35,0.04,0,zr,150.0,2.0
100,3,2458020.5,58020.0,18.10,0.50,0,zr,150.0,2.0
100,4,2458030.5,58030.0,18.25,0.05,0,zr,150.0,2.0
200,1,2458000.5,58000.0,19.90,0.02,0,zg,150.0,2.0
"""

def test_build_query_url_units():
    url = build_query_url(150.0, 2.0, "r", radius_arcsec=1.5)
    assert "CIRCLE+150.0+2.0+0.000417" in url
    assert "BANDNAME=r" in url

def test_parse_applies_band_and_magerr_cuts():
    curves = parse_lightcurve_csv(MOCK_CSV, band="r", max_magerr=0.2, label="1")
    assert len(curves) == 1 
    c = curves[0]
    assert c.oid == 100
    assert c.mjd.size == 3
    assert np.all(np.diff(c.mjd) > 0)  
    assert c.label == "1"

def test_best_curve_picks_most_epochs():
    curves = parse_lightcurve_csv(MOCK_CSV, band="r")
    assert best_curve(curves).oid == 100
    assert best_curve([]) is None

def test_ztf_dataset_matches_synthetic_interface(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(0)
    for oid in (1, 2):
        n = 300
        df = pd.DataFrame(
            {
                "mjd": np.cumsum(rng.uniform(1, 3, n)) + 58000,
                "mag": 18 + rng.normal(0, 0.1, n).astype(np.float32),
                "magerr": np.full(n, 0.03, np.float32),
                "label": np.full(n, float(oid % 2)),
            }
        )
        df.to_parquet(tmp_path / f"{oid}.parquet", index=False)
    from core.data.ztf_dataset import ZTFLightCurveDataset
    cfg = DataConfig(source="ztf", seq_len=128, ztf_cache_dir=str(tmp_path),
                     ztf_min_epochs=120, ztf_val_frac=0.5)
    ds = ZTFLightCurveDataset(cfg, split="train", base_seed=0)
    item = ds[0]
    assert set(item.keys()) == {"dt", "mag", "mag_err", "labels"}
    assert item["dt"].shape == (128,)
    assert item["mag"].shape == (128,)
    assert item["labels"].shape == (2,)
    assert float(item["dt"][0]) == 0.0 