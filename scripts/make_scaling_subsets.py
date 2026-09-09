from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def label_of(path: Path) -> float:
    df = pd.read_parquet(path, columns=["label"])
    return float(df["label"].iloc[0]) if len(df) else float("nan")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/ztf_cache")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 15000, 31000])
    ap.add_argument("--out-root", default="data/scale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cache = Path(args.cache).resolve()
    files = sorted(cache.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet in {cache}")
    labeled, unlabeled = [], []
    for f in files:
        (labeled if np.isfinite(label_of(f)) else unlabeled).append(f)
    print(f"[scale] {len(labeled)} labeled + {len(unlabeled)} unlabeled = {len(files)} total")
    rng = np.random.default_rng(args.seed)
    unlabeled = list(np.array(unlabeled)[rng.permutation(len(unlabeled))])  
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for size in sorted(args.sizes):
        n_unlab = max(0, size - len(labeled))
        if size < len(labeled):
            print(f"[scale] WARNING: size {size} < {len(labeled)} labeled; using labeled only")
        chosen = labeled + unlabeled[:n_unlab]  
        sub = out_root / f"n{size}"
        sub.mkdir(exist_ok=True)
        for old in sub.glob("*.parquet"):
            old.unlink()
        for f in chosen:
            (sub / f.name).symlink_to(Path(f).resolve())
        n_pos = sum(1 for f in labeled if label_of(f) == 1)
        print(f"[scale] {sub}: {len(chosen)} curves "
              f"({len(labeled)} labeled incl. {n_pos} positives + {n_unlab} unlabeled)")
    print(f"[scale] done. Point ztf_cache_dir at {out_root}/n<SIZE> for each scaling run.")

if __name__ == "__main__":
    main()