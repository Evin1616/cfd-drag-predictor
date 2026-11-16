"""Train the PointNet surrogate.

    python -m training.train --data data/synth_cars.npz --epochs 60

Writes weights/pointnet_cd_cl.pt plus a parity plot and metrics JSON in reports/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cfd_core.geometry import N_POINTS
from cfd_core.model import PointNetRegressor


class PointCloudSet(Dataset):
    def __init__(self, points, targets, train: bool):
        self.points, self.targets, self.train = points, targets, train

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        pts = self.points[i]
        if self.train:
            pts = pts[np.random.choice(len(pts), N_POINTS, replace=False)].copy()
            if np.random.rand() < 0.5:  # cars are left/right symmetric
                pts[:, 1] *= -1
                pts[:, 4] *= -1
            pts[:, :3] += np.random.normal(0, 0.002, (N_POINTS, 3)).astype(np.float32)
        else:
            pts = pts[:N_POINTS]
        return torch.from_numpy(pts), torch.from_numpy(self.targets[i])


def split_indices(n: int, seed: int = 0, val: float = 0.1, test: float = 0.1):
    idx = np.random.default_rng(seed).permutation(n)
    n_val, n_test = int(n * val), int(n * test)
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]


def evaluate(model, loader, mean, std):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for pts, y in loader:
            preds.append(model(pts).numpy())
            trues.append(y.numpy())
    p = np.concatenate(preds) * std + mean
    t = np.concatenate(trues) * std + mean
    return p, t


def metrics(p, t, names=("cd", "cl")):
    out = {}
    for k, name in enumerate(names):
        ok = ~np.isnan(t[:, k])  # e.g. designs without a lift value
        if not ok.any():
            continue
        err, tk = p[ok, k] - t[ok, k], t[ok, k]
        ss_res, ss_tot = (err**2).sum(), ((tk - tk.mean()) ** 2).sum()
        out[name] = {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": float(1 - ss_res / ss_tot),
            "mape_pct": float(100 * np.abs(err / tk).mean()) if name == "cd" else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_cars.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="weights/pointnet_cd_cl.pt")
    ap.add_argument("--dataset-name", default="synthetic Ahmed-trend car bodies (not CFD)")
    args = ap.parse_args()
    torch.manual_seed(0), np.random.seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    d = np.load(args.data)
    points = d["points"].astype(np.float32)
    y = np.stack([d["cd"], d["cl"]], axis=1).astype(np.float32)
    tr, va, te = split_indices(len(y))
    mean, std = np.nan_to_num(np.nanmean(y[tr], 0)), np.nan_to_num(np.nanstd(y[tr], 0), nan=1.0)
    yn = (y - mean) / std  # NaN targets stay NaN and are masked out of the loss

    loaders = {
        "train": DataLoader(PointCloudSet(points[tr], yn[tr], True), args.batch, shuffle=True, drop_last=True),
        "val": DataLoader(PointCloudSet(points[va], yn[va], False), 128),
        "test": DataLoader(PointCloudSet(points[te], yn[te], False), 128),
    }
    config = {"in_ch": 6, "n_out": 2, "width": 256, "dropout": 0.2}
    model = PointNetRegressor(**config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loaders["train"]))
    loss_fn = nn.SmoothL1Loss(beta=0.5, reduction="none")

    best, best_state, history = np.inf, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for pts, target in loaders["train"]:
            opt.zero_grad()
            ok = ~torch.isnan(target)
            loss = loss_fn(model(pts)[ok], target[ok]).mean()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(pts)
        p, t = evaluate(model, loaders["val"], mean, std)
        m = metrics(p, t)
        history.append({"epoch": epoch, "train_loss": running / len(tr), **{f"val_{k}_mae": v["mae"] for k, v in m.items()}})
        score = sum(m[k]["mae"] / std[i] for i, k in enumerate(("cd", "cl")) if k in m)
        if score < best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {running / len(tr):.4f}  val Cd MAE {m['cd']['mae']:.4f} "
              f"(R2 {m['cd']['r2']:.3f})  Cl MAE {m.get('cl', {}).get('mae', float('nan')):.4f}  {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    p, t = evaluate(model, loaders["test"], mean, std)
    test = metrics(p, t)
    print("TEST", json.dumps(test, indent=2))

    ext = points[tr, :, :3].max(1) - points[tr, :, :3].min(1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "config": config,
        "target_mean": mean.tolist(), "target_std": std.tolist(),
        "aspect_ranges": {"width/length": (float(ext[:, 1].min()), float(ext[:, 1].max())),
                          "height/length": (float(ext[:, 2].min()), float(ext[:, 2].max()))},
        "meta": {"dataset": args.dataset_name, "n_train": int(len(tr)), "test_metrics": test},
    }, args.out)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps({"dataset": args.dataset_name, "test": test, "history": history}, indent=2))
    np.savez(reports / "test_predictions.npz", pred=p, true=t)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
