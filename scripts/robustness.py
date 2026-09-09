from __future__ import annotations
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from core.anomaly import _oof_logistic

def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}

def probe_auc(emb: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    oof = _oof_logistic(emb, y, seed=seed, n_splits=5)
    return float(roc_auc_score(y, oof)), oof

def matched_control_auc(
    oof: np.ndarray,
    y: np.ndarray,
    median_mag: np.ndarray,
    n_bins: int = 10,
    n_resamples: int = 50,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    edges = np.quantile(median_mag[pos], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pos_bins = np.digitize(median_mag[pos], edges) - 1
    neg_bins = np.digitize(median_mag[neg], edges) - 1
    aucs = []
    for _ in range(n_resamples):
        picked: list[int] = []
        ok = True
        for b in range(n_bins):
            n_pos_b = int((pos_bins == b).sum())
            pool = neg[neg_bins == b]
            if n_pos_b == 0:
                continue
            if len(pool) < n_pos_b:
                ok = False 
                break
            picked.extend(rng.choice(pool, size=n_pos_b, replace=False).tolist())
        if not ok or not picked:
            continue
        idx = np.concatenate([pos, np.asarray(picked)])
        aucs.append(roc_auc_score(y[idx], oof[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))

def main() -> None:
    ap = argparse.ArgumentParser(description="Confound + stability checks for the probe result.")
    ap.add_argument("--features", default="runs/ztf/features.npz")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    d = load(args.features)
    labels, emb, mmag = d["labels"], d["embeddings"], d["median_mag"]
    mask = np.isfinite(labels)
    y = labels[mask].astype(int)
    emb_l, mmag_l = emb[mask], mmag[mask]
    print(f"[robust] labeled: {len(y)} ({y.sum()} positives, base rate {y.mean():.4f})")
    auc_bright = roc_auc_score(y, mmag_l)
    auc_bright = max(auc_bright, 1 - auc_bright)
    print(f"\n[robust] 1) median-brightness ALONE:      AUC = {auc_bright:.4f}")
    print("          (how far a trivial 'faint=CLAGN' rule gets on this label set)")
    aucs, oofs = [], []
    for s in range(args.n_seeds):
        a, oof = probe_auc(emb_l, y, seed=s)
        aucs.append(a)
        oofs.append(oof)
    print(f"\n[robust] 2) probe AUC over {args.n_seeds} seeds:      "
          f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}   (paper number)")
    m_aucs = []
    for s, oof in enumerate(oofs):
        m, _ = matched_control_auc(oof, y, mmag_l, seed=s)
        if np.isfinite(m):
            m_aucs.append(m)
    if m_aucs:
        print(f"\n[robust] 3) brightness-MATCHED probe AUC: "
              f"{np.mean(m_aucs):.4f} +/- {np.std(m_aucs):.4f}")
        drop = np.mean(aucs) - np.mean(m_aucs)
        print(f"          drop vs unmatched: {drop:+.4f}")
        print("          interpretation: signal remaining after the brightness axis is")
        print("          removed. Small drop -> variability structure is the driver;")
        print("          large drop -> the result was mostly the selection function.")
    else:
        print("\n[robust] 3) matched AUC: not enough brightness-matched controls; "
              "widen bins (--n-bins) or add fainter controls.")

if __name__ == "__main__":
    main()