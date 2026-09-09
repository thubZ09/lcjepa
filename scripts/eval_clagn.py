from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from core.anomaly import (
    Features,
    _rank01,
    _shape_features,
    combined_score,
    evaluate,
    isolation_forest_scores,
    supervised_score,
)
from core.config import Config, _build
from core.data.synthetic import collate
from core.models.lcjepa import LCJEPA
from core.device import configure_backend, select_device

def load_checkpoint(path: str, device: torch.device) -> tuple[LCJEPA, Config]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _build(Config, ckpt["config"])
    model = LCJEPA(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    benign = [k for k in missing if k.startswith("recon_head")]
    bad_missing = [k for k in missing if not k.startswith("recon_head")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={bad_missing} unexpected={list(unexpected)}"
        )
    if benign:
        print(f"[eval] note: {benign} not in this (older) checkpoint — unused at eval, OK")
    model.eval()
    return model, cfg

def load_cache(cache_dir: str, min_epochs: int) -> tuple[list[dict], list[int]]:
    import pandas as pd
    files = sorted(Path(cache_dir).glob("*.parquet"))
    records, oids = [], []
    n_short = 0
    for f in files:
        df = pd.read_parquet(f)
        if len(df) < min_epochs:
            n_short += 1
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
    print(f"[eval] loaded {len(records)} curves ({n_short} below {min_epochs} epochs dropped)")
    return records, oids

def full_curve_shape(records: list[dict]) -> np.ndarray:
    out = np.zeros((len(records), 2), dtype=np.float32)
    for i, r in enumerate(records):
        mag = r["mag"]
        m = (mag - mag.mean()) / (mag.std() + 1e-6)
        out[i] = _shape_features(m.astype(np.float32))
    return out

class _WindowBatch(Dataset):
    def __init__(self, records: list[dict], seq_len: int, n_windows: int):
        self.items: list[tuple[int, int]] = []  
        self.records = records
        self.seq_len = seq_len
        for i, r in enumerate(records):
            n = r["mjd"].size
            if n <= seq_len:
                starts = [0]
            else:
                starts = np.unique(
                    np.linspace(0, n - seq_len, num=min(n_windows, n - seq_len + 1))
                    .astype(int)
                ).tolist()
            for s in starts:
                self.items.append((i, s))
    def __len__(self) -> int:
        return len(self.items)
    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        i, s = self.items[k]
        r = self.records[i]
        sl = slice(s, s + self.seq_len)
        mjd, mag, magerr = r["mjd"][sl], r["mag"][sl], r["magerr"][sl]
        if mjd.size < self.seq_len:  
            pad = self.seq_len - mjd.size
            mjd = np.concatenate([mjd, np.full(pad, mjd[-1])])
            mag = np.concatenate([mag, np.full(pad, mag[-1])])
            magerr = np.concatenate([magerr, np.full(pad, magerr[-1])])
        dt = (mjd - mjd[0]).astype(np.float32)
        mu, sd = float(mag.mean()), float(mag.std() + 1e-6)
        return {
            "dt": torch.from_numpy(dt),
            "mag": torch.from_numpy(((mag - mu) / sd).astype(np.float32)),
            "mag_err": torch.from_numpy((magerr / sd).astype(np.float32)),
            "labels": torch.tensor([float(i), np.nan]),  # smuggle record idx
        }

@torch.no_grad()
def multiwindow_features(
    model: LCJEPA,
    records: list[dict],
    seq_len: int,
    n_windows: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ds = _WindowBatch(records, seq_len, n_windows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    d_emb = None
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    maxes: dict[int, float] = {}
    for batch in loader:
        idxs = batch["labels"][:, 0].numpy().astype(int)
        b = {k: v.to(device) for k, v in batch.items()}
        emb = model.embed_sequence(b).cpu().numpy()
        sur = model.rollout_surprise(b).cpu().numpy()
        d_emb = emb.shape[1]
        for j, i in enumerate(idxs):
            sums[i] = sums.get(i, np.zeros(d_emb)) + emb[j]
            counts[i] = counts.get(i, 0) + 1
            maxes[i] = max(maxes.get(i, -np.inf), float(sur[j]))
    n = len(records)
    embeddings = np.stack([sums[i] / counts[i] for i in range(n)])
    surprise = np.array([maxes[i] for i in range(n)], dtype=np.float32)
    return embeddings, surprise

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CLAGN anomaly detection (v2).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="runs/ztf/candidates.csv")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--n-windows", type=int, default=5,
                    help="Model-scored windows per curve (max-aggregated surprise).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="Skip loading weights: probe a RANDOM-INIT encoder (ablation).")
    args = ap.parse_args()
    device = select_device(args.device)
    configure_backend(device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    if args.random_init:
        from core.models.lcjepa import LCJEPA as _L
        model = _L(cfg).to(device); model.eval()
        print('[eval] RANDOM-INIT ablation: weights NOT loaded')
    min_epochs = max(cfg.data.ztf_min_epochs, cfg.data.seq_len)
    records, oids = load_cache(cfg.data.ztf_cache_dir, min_epochs)
    labels = np.array([r["label"] for r in records], dtype=np.float32)
    print(f"[eval] multi-window model scoring on {device.type} "
          f"({args.n_windows} windows/curve)...")
    embeddings, surprise = multiwindow_features(
        model, records, cfg.data.seq_len, args.n_windows, cfg.optim.batch_size, device
    )
    shape = full_curve_shape(records)
    feats = Features(embeddings=embeddings, surprise=surprise, labels=labels, shape=shape)
    median_mag = np.array([float(np.median(r["mag"])) for r in records], dtype=np.float32)
    feat_path = Path(args.out).parent / "features.npz"
    feat_path.parent.mkdir(parents=True, exist_ok=True)  
    np.savez_compressed(
        feat_path,
        embeddings=embeddings, surprise=surprise, shape=shape,
        labels=labels, oids=np.asarray(oids, dtype=np.int64), median_mag=median_mag,
    )
    print(f"[eval] saved features -> {feat_path}")
    print("\n[eval]component AUCs (labeled subset")
    from sklearn.metrics import roc_auc_score
    mask = np.isfinite(labels)
    y = labels[mask].astype(int)
    if len(np.unique(y)) == 2:
        iso = _rank01(isolation_forest_scores(embeddings))[mask]
        comps = {
            "IF(embeddings)": iso,
            "surprise(max, signed)": surprise[mask],
            "half-mean step (full curve)": shape[mask, 0],
            "CUSUM step (full curve)": shape[mask, 1],
        }
        for name, s in comps.items():
            print(f"  {name:>30s}: AUC={roc_auc_score(y, s):.4f}")
        print("  (CUSUM-full is the 'does photometric signal exist at all' diagnostic;")
        print("   an AUC near 0.5 there means transitions are largely outside ZTF's window.)")
    unsup = combined_score(feats)
    metrics_unsup = evaluate(unsup, labels)
    sup = supervised_score(feats)
    metrics_sup = evaluate(np.nan_to_num(sup, nan=-1.0), labels)
    from core.anomaly import baseline_probe_score, embedding_probe_score
    probe = embedding_probe_score(feats)
    metrics_probe = evaluate(np.nan_to_num(probe, nan=-1.0), labels)
    baseline = baseline_probe_score(records, labels)
    metrics_base = evaluate(np.nan_to_num(baseline, nan=-1.0), labels)
    print("\n[eval]unsupervised combined")
    for k, v in metrics_unsup.items():
        print(f"  {k:>20s}: {v:.4f}")
    print("[eval]label-calibrated component stack (OOF)")
    for k, v in metrics_sup.items():
        print(f"  {k:>20s}: {v:.4f}")
    print("[eval] LINEAR PROBE on frozen embeddings (OOF) ")
    for k, v in metrics_probe.items():
        print(f"  {k:>20s}: {v:.4f}")
    print("[eval] raw-feature baseline, same protocol (OOF)")
    for k, v in metrics_base.items():
        print(f"  {k:>20s}: {v:.4f}")
    base = float(np.nanmean(labels))
    print(f"  {'base rate':>20s}: {base:.4f}  (precision@k must beat this)")
    if "roc_auc" in metrics_probe and "roc_auc" in metrics_base:
        d = metrics_probe["roc_auc"] - metrics_base["roc_auc"]
        print(f"\n[eval] representation vs hand-crafted features: "
              f"probe {metrics_probe['roc_auc']:.4f} vs baseline "
              f"{metrics_base['roc_auc']:.4f} (Δ={d:+.4f})")
    score = np.where(np.isfinite(probe), probe, unsup)
    order = np.argsort(-score)[: args.topk]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "oid", "score", "label"])
        w.writeheader()
        for rank, i in enumerate(order, start=1):
            w.writerow({
                "rank": rank,
                "oid": oids[i],
                "score": f"{float(score[i]):.5f}",
                "label": int(labels[i]) if np.isfinite(labels[i]) else "",
            })
    print(f"\n[eval] wrote top-{args.topk} candidates -> {out}")

if __name__ == "__main__":
    main()