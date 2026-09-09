from __future__ import annotations
import argparse
import time
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from core.config import Config
from core.data.drw import sample_params, simulate_clagn, simulate_drw
from core.data.synthetic import SyntheticDRWDataset, collate
from core.anomaly import (
    Features,
    _rank01,
    combined_score,
    extract_features,
    isolation_forest_scores,
    supervised_score,
)
from core.probe import probe_physics
from core.models.lcjepa import LCJEPA
from core.device import configure_backend, select_device
from core.seed import seed_everything

class MixedAnomalyDataset(Dataset):
    def __init__(self, cfg, n_each: int, base_seed: int):
        self._items: list[dict] = []
        self._labels: list[int] = []
        d = cfg.data
        for i in range(n_each):
            rng = np.random.default_rng(base_seed + i)
            lt, ls = sample_params(d.log_tau_range, d.log_sigma_range, rng)
            normal = simulate_drw(d.seq_len, lt, ls, d.cadence_days, d.cadence_jitter,
                                  d.season_gap_frac, d.phot_err, rng)
            self._items.append(self._norm(normal, d.seq_len)); self._labels.append(0)
            rng2 = np.random.default_rng(base_seed + 10_000_000 + i)
            lt, ls = sample_params(d.log_tau_range, d.log_sigma_range, rng2)
            clagn = simulate_clagn(d.seq_len, lt, ls, d.cadence_days, d.cadence_jitter,
                                   d.season_gap_frac, d.phot_err, rng2)
            self._items.append(self._norm(clagn, d.seq_len)); self._labels.append(1)

    @staticmethod
    def _norm(lc, seq_len):
        dt = (lc.t - lc.t[0]).astype(np.float32)
        m = (lc.mag - lc.mag.mean()) / (lc.mag.std() + 1e-6)
        return {"dt": dt, "mag": m.astype(np.float32), "mag_err": lc.mag_err.astype(np.float32)}
    def labels(self) -> np.ndarray:
        return np.asarray(self._labels)
    def __len__(self):
        return len(self._items)
    def __getitem__(self, i):
        it = self._items[i]
        return {
            "dt": torch.from_numpy(it["dt"]),
            "mag": torch.from_numpy(it["mag"]),
            "mag_err": torch.from_numpy(it["mag_err"]),
            "labels": torch.tensor([float(self._labels[i]), np.nan], dtype=torch.float32),
        }

def raw_feature_auc(ds: MixedAnomalyDataset, seed: int) -> float:
    feats = []
    for i in range(len(ds)):
        it = ds[i]
        mag = it["mag"].numpy()
        dmag = np.diff(mag)
        feats.append([
            mag.std(),
            np.median(np.abs(mag - np.median(mag))),    
            np.mean(np.abs(dmag)),                      
            float(np.mean((mag - mag.mean()) ** 3)),   
            float(mag.max() - mag.min()),             
        ])
    X = np.asarray(feats)
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(X)
    return float(roc_auc_score(ds.labels(), -iso.score_samples(X)))

@torch.no_grad()
def _extract_full(model, loader, device) -> Features:
    return extract_features(model, loader, device)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test-each", type=int, default=400)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--enc-depth", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=160)
    ap.add_argument("--ema-base", type=float, default=0.99)      
    ap.add_argument("--lr", type=float, default=1.5e-3) 
    args = ap.parse_args()
    seed_everything(0)
    device = select_device(args.device)
    configure_backend(device)
    cfg = Config()
    cfg.data.n_jobs = 4
    cfg.data.seq_len = args.seq_len
    cfg.model.d_model = args.d_model
    cfg.model.enc_depth = args.enc_depth
    cfg.optim.ema_base = args.ema_base       
    cfg.optim.batch_size = 128
    print(f"[validate] device={device.type} epochs={args.epochs}")
    train_ds = SyntheticDRWDataset(cfg.data, args.n_train, base_seed=1)
    train_loader = DataLoader(train_ds, batch_size=cfg.optim.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_ds = SyntheticDRWDataset(cfg.data, 400, base_seed=5_000_000)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate)
    model = LCJEPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[validate] params={n_params/1e6:.2f}M")
    r2_init = probe_physics(model, train_loader, val_loader, device)["r2_log_tau"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.04, betas=(0.9, 0.95))
    total = args.epochs * len(train_loader)
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            mom = cfg.optim.ema_base + (1.0 - cfg.optim.ema_base) * step / max(total, 1)
            model.update_target(mom)
            step += 1
        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1:02d} loss {out.loss.item():.3f} "
                  f"pred {out.pred_loss.item():.3f} roll {out.rollout_loss.item():.3f}")
    print(f"[validate] trained in {time.time()-t0:.1f}s")
    r2_trained = probe_physics(model, train_loader, val_loader, device)["r2_log_tau"]
    test_ds = MixedAnomalyDataset(cfg, args.n_test_each, base_seed=2_000)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)
    feats = _extract_full(model, test_loader, device)
    y = test_ds.labels()
    auc_unsup = roc_auc_score(y, combined_score(feats))          
    auc_sup = roc_auc_score(y, supervised_score(feats))           
    auc_surprise = roc_auc_score(y, feats.surprise)          
    auc_iso = roc_auc_score(y, _rank01(isolation_forest_scores(feats.embeddings)))
    auc_step = roc_auc_score(y, feats.shape[:, 1])
    auc_raw = raw_feature_auc(test_ds, seed=0)
    print(f"1) Representation     : R2(log_tau) init={r2_init:+.3f} -> trained={r2_trained:+.3f}"
          f"   {'PASS' if r2_trained > r2_init + 0.03 else 'CHECK (needs full run)'}")
    print(f"2) Components         : IF={auc_iso:.3f}  surprise(signed)={auc_surprise:.3f}  "
          f"step-cusum={auc_step:.3f}")
    print(f"3) Unsup combined     : ROC-AUC={auc_unsup:.3f}"
          f"   {'PASS' if auc_unsup > 0.65 else 'CHECK'}")
    print(f"4) Label-calibrated   : ROC-AUC(OOF)={auc_sup:.3f}"
          f"   {'PASS' if auc_sup > 0.70 else 'CHECK'}")
    print(f"5) Beats raw features : calibrated={auc_sup:.3f} vs raw={auc_raw:.3f}"
          f"   {'PASS' if auc_sup > auc_raw else 'CHECK'}")

if __name__ == "__main__":
    main()