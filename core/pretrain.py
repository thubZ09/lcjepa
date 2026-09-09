from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core.config import Config
from core.data.synthetic import SyntheticDRWDataset, collate
from core.probe import probe_physics
from core.models.lcjepa import LCJEPA
from core.device import configure_backend, empty_cache, select_device
from core.seed import seed_everything

def _cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))

def _ema_momentum(step: int, total: int, base: float, final: float) -> float:
    return final - (final - base) * (math.cos(math.pi * step / max(total, 1)) + 1) / 2

def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    if cfg.data.source == "ztf":
        from core.data.ztf_dataset import ZTFLightCurveDataset
        train_ds: object = ZTFLightCurveDataset(cfg.data, split="train", base_seed=1_000)
        val_ds: object = ZTFLightCurveDataset(cfg.data, split="val", base_seed=1_000)
    else:
        train_ds = SyntheticDRWDataset(cfg.data, cfg.data.n_train, base_seed=1_000)
        val_ds = SyntheticDRWDataset(cfg.data, cfg.data.n_val, base_seed=9_000_000)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size, shuffle=True,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.optim.batch_size, shuffle=False, collate_fn=collate,
    )
    return train_loader, val_loader

def train(cfg: Config) -> dict[str, float]:
    seed_everything(cfg.run.seed)
    device = select_device(cfg.run.device)
    configure_backend(device)
    out_dir = Path(cfg.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    print(f"[lcjepa] device={device.type}")
    train_loader, val_loader = build_loaders(cfg)
    print(f"[lcjepa] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    model = LCJEPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lcjepa] trainable params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, betas=(0.9, 0.95),
    )
    total_steps = cfg.optim.epochs * len(train_loader)
    step = 0
    history: dict[str, float] = {}
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lr = _cosine_lr(step, total_steps, cfg.optim.warmup_epochs * len(train_loader), cfg.optim.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            out = model(batch)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            mom = _ema_momentum(step, total_steps, cfg.optim.ema_base, cfg.optim.ema_final)
            model.update_target(mom)
            if step % cfg.run.log_every == 0:
                print(
                    f"e{epoch:02d} s{step:05d} lr{lr:.2e} "
                    f"loss {out.loss.item():.4f} | pred {out.pred_loss.item():.4f} "
                    f"roll {out.rollout_loss.item():.4f} var {out.vic_var.item():.4f} "
                    f"cov {out.vic_cov.item():.4f}"
                )
            step += 1
        empty_cache(device)
        run_probe = cfg.data.source == "synthetic" and (
            (epoch + 1) % cfg.run.probe_every == 0 or epoch == cfg.optim.epochs - 1
        )
        if run_probe:
            probe = probe_physics(model, train_loader, val_loader, device)
            history = probe
            print(
                f"[probe e{epoch:02d}] R2 tau={probe['r2_log_tau']:.3f} "
                f"sigma={probe['r2_log_sigma']:.3f} mean={probe['r2_mean']:.3f} "
                f"({time.time()-t0:.1f}s/epoch)"
            )
        elif cfg.data.source == "ztf":
            print(f"[e{epoch:02d}] done ({time.time()-t0:.1f}s/epoch)")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out_dir / "lcjepa.pt")
    print(f"[lcjepa] saved checkpoint -> {out_dir/'lcjepa.pt'}")
    return history

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain LC-JEPA on synthetic DRW light curves.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/mps/cuda).")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.device is not None:
        cfg.run.device = args.device
    train(cfg)

if __name__ == "__main__":
    main()