from __future__ import annotations
import argparse
import json
from pathlib import Pat
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def load_run(spec: str) -> tuple[str, dict]:
    name, path = spec.split("=", 1)
    z = np.load(path)
    return name, {k: z[k] for k in z.files}

def _oof_auc(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5, C: float = 0.1) -> float:
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof))

def _subset_auc(X: np.ndarray, y: np.ndarray, n_pos: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if n_pos >= len(pos):
        return _oof_auc(X, y, seed)
    n_neg = min(len(neg), n_pos * (len(neg) // len(pos)))  
    tr_pos = rng.choice(pos, n_pos, replace=False)
    tr_neg = rng.choice(neg, n_neg, replace=False)
    tr = np.concatenate([tr_pos, tr_neg])
    te = np.setdiff1d(np.arange(len(y)), tr)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    clf.fit(sc.transform(X[tr]), y[tr])
    s = clf.predict_proba(sc.transform(X[te]))[:, 1]
    if len(np.unique(y[te])) < 2:
        return float("nan")
    return float(roc_auc_score(y[te], s))

def label_efficiency(runs: dict, ks, seeds, raw_feats=None) -> dict:
    out = {}
    for name, d in runs.items():
        mask = np.isfinite(d["labels"])
        y = d["labels"][mask].astype(int)
        X = d["embeddings"][mask]
        curve = {}
        for k in ks:
            if k > int(y.sum()):
                continue
            aucs = [_subset_auc(X, y, k, s) for s in seeds]
            aucs = [a for a in aucs if np.isfinite(a)]
            curve[k] = (float(np.mean(aucs)), float(np.std(aucs)))
        out[name] = curve
    return out

def data_scaling(runs: dict, seeds) -> dict:
    out = {}
    for name, d in runs.items():
        mask = np.isfinite(d["labels"])
        y = d["labels"][mask].astype(int)
        X = d["embeddings"][mask]
        aucs = [_oof_auc(X, y, s) for s in seeds]
        out[name] = (float(np.mean(aucs)), float(np.std(aucs)))
    return out

def pca_rank(runs: dict, ranks, seed: int = 0) -> dict:
    from sklearn.decomposition import PCA
    out = {}
    for name, d in runs.items():
        mask = np.isfinite(d["labels"])
        y = d["labels"][mask].astype(int)
        X = d["embeddings"][mask]
        curve = {}
        for r in ranks:
            r = min(r, X.shape[1], X.shape[0] - 1)
            Xr = PCA(n_components=r, random_state=seed).fit_transform(
                StandardScaler().fit_transform(X)
            )
            curve[r] = _oof_auc(Xr, y, seed)
        out[name] = curve
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="name=path/to/features.npz (repeatable).")
    ap.add_argument("--out", default="runs/analysis")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    runs = dict(load_run(s) for s in args.run)
    seeds = list(range(args.n_seeds))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    n_pos = int(np.isfinite(next(iter(runs.values()))["labels"]).sum())
    print(f"[analyze] runs: {list(runs)} | positives available: "
          f"{int((next(iter(runs.values()))['labels'] == 1).sum())}")
    ks = [10, 25, 50, 100, 200, 373]
    le = label_efficiency(runs, ks, seeds)
    print("\n[analyze](probe AUC, mean +/- std)")
    header = "  k     " + "".join(f"{n:>18s}" for n in runs)
    print(header)
    for k in ks:
        row = f"  {k:<5d} "
        for name in runs:
            if k in le[name]:
                m, s = le[name][k]
                row += f"   {m:.3f}±{s:.3f}   "
            else:
                row += f"{'—':>18s}"
        print(row)
    ds = data_scaling(runs, seeds)
    print("\n[analyze](data-scaling anchor)")
    for name, (m, s) in ds.items():
        print(f"  {name:>18s}: {m:.4f} ± {s:.4f}")
    pr = pca_rank(runs, [2, 4, 8, 16, 32, 64, 128, 256])
    print("\n[analyze]PCA-RANK (probe AUC vs # components)")
    for name, curve in pr.items():
        best = max(curve.values())
        knee = min(r for r, a in curve.items() if a >= 0.98 * best)
        print(f"  {name:>18s}: reaches 98% of max ({best:.3f}) by {knee} components")
    results = {"label_efficiency": le, "data_scaling": ds, "pca_rank": pr, "ks": ks}
    (outdir / "analysis.json").write_text(json.dumps(results, indent=2))
    print(f"\n[analyze] saved -> {outdir/'analysis.json'} (feeds make_figures.py)")

if __name__ == "__main__":
    main()