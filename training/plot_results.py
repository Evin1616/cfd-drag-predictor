"""Parity plot of test-set predictions vs. labels, for the README / LinkedIn post.

    python -m training.plot_results
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORTS = Path("reports")


def main() -> None:
    d = np.load(REPORTS / "test_predictions.npz")
    m = json.loads((REPORTS / "metrics.json").read_text())["test"]
    targets = [(k, name, label, color) for k, (name, label, color) in enumerate(
        (("cd", "Drag coefficient $C_d$", "#d9534f"), ("cl", "Lift coefficient $C_l$", "#4a90d9"))) if name in m]
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets) + 0.6, 5.0), squeeze=False)
    for ax, (k, name, label, color) in zip(axes[0], targets):
        ok = ~np.isnan(d["true"][:, k])
        t, p = d["true"][ok, k], d["pred"][ok, k]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=1, ls="--", zorder=1)
        ax.scatter(t, p, s=14, alpha=0.6, color=color, edgecolor="none", zorder=2)
        ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel=f"true {label}",
               ylabel=f"predicted {label}", aspect="equal")
        mape = f"   MAPE = {m[name]['mape_pct']:.1f}%" if m[name].get("mape_pct") is not None else ""
        ax.set_title(f"$R^2$ = {m[name]['r2']:.3f}   MAE = {m[name]['mae']:.4f}{mape}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    title = json.loads((REPORTS / "metrics.json").read_text()).get("dataset", "")
    fig.suptitle(f"PointNet surrogate, held-out test set ({len(d['true'])} unseen shapes)\n{title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "parity.png", dpi=160)
    print("saved", REPORTS / "parity.png")


if __name__ == "__main__":
    main()
