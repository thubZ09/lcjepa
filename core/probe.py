from __future__ import annotations
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from core.models.lcjepa import LCJEPA

@torch.no_grad()
def _collect(model: LCJEPA, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        feats.append(model.embed_sequence(batch).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)

def probe_physics(
    model: LCJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
) -> dict[str, float]:
    xtr, ytr = _collect(model, train_loader, device)
    xva, yva = _collect(model, val_loader, device)
    results: dict[str, float] = {}
    names = ["log_tau", "log_sigma"]
    r2s = []
    for j, name in enumerate(names):
        reg = Ridge(alpha=alpha).fit(xtr, ytr[:, j])
        r2 = float(r2_score(yva[:, j], reg.predict(xva)))
        results[f"r2_{name}"] = r2
        r2s.append(r2)
    results["r2_mean"] = float(np.mean(r2s))
    return results