from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
from core.models.lcjepa import LCJEPA

@dataclass
class Features:
    embeddings: np.ndarray
    surprise: np.ndarray    
    labels: np.ndarray     
    shape: np.ndarray | None = None 

def _shape_features(mag: np.ndarray) -> tuple[float, float]:
    n = mag.size
    h = n // 2
    half_diff = abs(float(mag[:h].mean() - mag[h:].mean()))
    cusum = np.cumsum(mag - mag.mean())
    max_cusum = float(np.max(np.abs(cusum))) / max(n, 1)
    return half_diff, max_cusum

@torch.no_grad()
def extract_features(model: LCJEPA, loader: DataLoader, device: torch.device) -> Features:
    model.eval()
    embs, surps, labs, shapes = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        embs.append(model.embed_sequence(batch).cpu().numpy())
        surps.append(model.rollout_surprise(batch).cpu().numpy())
        labs.append(batch["labels"][:, 0].cpu().numpy())
        mags = batch["mag"].cpu().numpy()
        shapes.append(np.array([_shape_features(m) for m in mags], dtype=np.float32))
    return Features(
        embeddings=np.concatenate(embs),
        surprise=np.concatenate(surps),
        labels=np.concatenate(labs),
        shape=np.concatenate(shapes),
    )

def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)

def isolation_forest_scores(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    iso.fit(embeddings)
    return -iso.score_samples(embeddings)

def _component_scores(feats: Features, seed: int) -> np.ndarray:
    iso = _rank01(isolation_forest_scores(feats.embeddings, seed=seed))
    sur = _rank01(feats.surprise)
    cols = [iso, sur]
    if feats.shape is not None:
        cols.append(_rank01(feats.shape[:, 0]))
        cols.append(_rank01(feats.shape[:, 1]))
    return np.column_stack(cols)

def combined_score(feats: Features, w_surprise: float = 0.5, seed: int = 0) -> np.ndarray:
    comp = _component_scores(feats, seed)
    iso = comp[:, 0]
    sur_two_sided = np.abs(comp[:, 1] - 0.5) * 2.0
    parts = [iso, sur_two_sided]
    if feats.shape is not None:
        parts.extend([comp[:, 2], comp[:, 3]])
    return np.mean(np.column_stack(parts), axis=1)

def supervised_score(feats: Features, seed: int = 0, n_splits: int = 5) -> np.ndarray:
    mask = np.isfinite(feats.labels)
    y = feats.labels[mask].astype(int)
    X = _component_scores(feats, seed)[mask]
    oof = _oof_logistic(X, y, seed, n_splits)
    full = np.full(len(feats.labels), np.nan)
    full[mask] = oof
    return full

def _oof_logistic(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    skf = StratifiedKFold(
        n_splits=min(n_splits, int(y.sum()), int((1 - y).sum())),
        shuffle=True, random_state=seed,
    )
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        clf.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return oof

def embedding_probe_score(feats: Features, seed: int = 0, n_splits: int = 5) -> np.ndarray:
    mask = np.isfinite(feats.labels)
    y = feats.labels[mask].astype(int)
    X = feats.embeddings[mask]
    oof = _oof_logistic(X, y, seed, n_splits)
    full = np.full(len(feats.labels), np.nan)
    full[mask] = oof
    return full

def raw_feature_matrix(records: list[dict]) -> np.ndarray:
    rows = []
    for r in records:
        mag = np.asarray(r["mag"], dtype=np.float64)
        dmag = np.diff(mag)
        m_norm = ((mag - mag.mean()) / (mag.std() + 1e-6)).astype(np.float32)
        half_diff, cusum = _shape_features(m_norm)
        centered = mag - mag.mean()
        std = mag.std() + 1e-12
        rows.append([
            mag.std(),
            float(np.median(np.abs(mag - np.median(mag)))),
            float(np.mean(np.abs(dmag))) if dmag.size else 0.0,
            float(np.mean(centered**3) / std**3),
            float(mag.max() - mag.min()),
            float(np.median(mag)),
            half_diff,
            cusum,
        ])
    return np.asarray(rows, dtype=np.float64)

def baseline_probe_score(
    records: list[dict], labels: np.ndarray, seed: int = 0, n_splits: int = 5
) -> np.ndarray:
    mask = np.isfinite(labels)
    y = labels[mask].astype(int)
    X = raw_feature_matrix([r for r, m in zip(records, mask) if m])
    oof = _oof_logistic(X, y, seed, n_splits)
    full = np.full(len(labels), np.nan)
    full[mask] = oof
    return full

def evaluate(scores: np.ndarray, labels: np.ndarray, ks=(20, 50, 100)) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    out: dict[str, float] = {}
    mask = np.isfinite(labels)
    if mask.sum() >= 2 and len(np.unique(labels[mask])) == 2:
        y, s = labels[mask].astype(int), scores[mask]
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))
        order = np.argsort(-s)
        for k in ks:
            kk = min(k, len(s))
            out[f"precision_at_{k}"] = float(y[order[:kk]].mean())
    return out

def ranked_candidates(
    oids: list[int], scores: np.ndarray, labels: np.ndarray, top: int = 100
) -> list[dict]:
    order = np.argsort(-scores)[:top]
    rows = []
    for rank, i in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "oid": oids[i] if i < len(oids) else -1,
                "score": float(scores[i]),
                "label": float(labels[i]) if np.isfinite(labels[i]) else "",
            }
        )
    return rows