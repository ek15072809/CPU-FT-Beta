"""Training-history plots.

The trainer records a compact JSON history.  This module turns that history
into standalone PNG files so the graphs survive after the console window is
closed and can be viewed without the Web UI.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_history(run_dir: Path) -> dict:
    path = Path(run_dir) / "history.json"
    if not path.exists():
        return {"train": [], "val": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"train": [], "val": []}
    if not isinstance(data, dict):
        return {"train": [], "val": []}
    return {
        "train": list(data.get("train") or []),
        "val": list(data.get("val") or []),
    }


def _values(rows, key, default=None):
    out = []
    for row in rows:
        value = row.get(key, default)
        try:
            out.append(float(value) if value is not None else float("nan"))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _empty_plot(plt, path: Path, title: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_title(title)
    ax.text(0.5, 0.5, note, ha="center", va="center",
            transform=ax.transAxes, color="#666666")
    ax.set_axis_off()
    _save(fig, path)


def generate_training_plots(run_dir, out_dir=None, summary=None) -> dict[str, str]:
    """Write lr.png, train_loss.png, grad_norm.png, val_ce.png and an overview.

    Returns a mapping of plot name to absolute path.  Plotting is best-effort:
    a missing optional dependency never fails training.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "plots"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    history = _load_history(run_dir)
    train = history["train"]
    val = history["val"]
    summary = summary or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    train_steps = [int(r.get("step") or 0) for r in train]
    lr = _values(train, "lr")
    loss = _values(train, "true")
    ema = _values(train, "ema")
    grad = _values(train, "grad_norm")
    val_steps = [int(r.get("step") or 0) for r in val]
    val_ce = _values(val, "val_ce")
    probe = _values(val, "probe_ce")
    best_step = summary.get("val_ce_best_step")
    restarts = list(summary.get("restarts") or [])

    # Every plot uses the same restrained palette as the UI.
    colors = {
        "lr": "#0b6e99",
        "loss": "#b45309",
        "ema": "#64748b",
        "grad": "#7c3aed",
        "val": "#047857",
        "probe": "#94a3b8",
        "best": "#c2410c",
        "restart": "#b91c1c",
    }

    def mark_events(ax):
        for event in restarts:
            step = event.get("from_step")
            if step is not None:
                ax.axvline(float(step), color=colors["restart"],
                           alpha=0.28, linestyle="--", linewidth=0.9)
        if best_step is not None:
            ax.axvline(float(best_step), color=colors["best"],
                       alpha=0.30, linestyle=":", linewidth=1.0)

    if train_steps and lr:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(train_steps, lr, color=colors["lr"], linewidth=1.5)
        ax.set_title("Learning rate")
        ax.set_xlabel("step")
        ax.set_ylabel("LR")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.18)
        mark_events(ax)
        p = out_dir / "lr.png"
        _save(fig, p)
        paths["lr"] = str(p.resolve())
    else:
        p = out_dir / "lr.png"
        _empty_plot(plt, p, "Learning rate", "No LR history")
        paths["lr"] = str(p.resolve())

    if train_steps and loss:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(train_steps, loss, color=colors["loss"], alpha=0.45,
                linewidth=0.8, label="step loss")
        if ema and any(v == v for v in ema):
            ax.plot(train_steps, ema, color=colors["ema"], linewidth=1.6,
                    label="EMA")
        ax.set_title("Training loss (sampled ASCOT objective)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
        mark_events(ax)
        p = out_dir / "train_loss.png"
        _save(fig, p)
        paths["train_loss"] = str(p.resolve())
    else:
        p = out_dir / "train_loss.png"
        _empty_plot(plt, p, "Training loss", "No training history")
        paths["train_loss"] = str(p.resolve())

    if train_steps and grad and any(v == v for v in grad):
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(train_steps, grad, color=colors["grad"], linewidth=1.0)
        ax.set_title("Gradient norm")
        ax.set_xlabel("step")
        ax.set_ylabel("grad norm")
        ax.grid(alpha=0.18)
        mark_events(ax)
        p = out_dir / "grad_norm.png"
        _save(fig, p)
        paths["grad_norm"] = str(p.resolve())
    else:
        p = out_dir / "grad_norm.png"
        _empty_plot(plt, p, "Gradient norm",
                    "Gradient norm is recorded from v9 runs")
        paths["grad_norm"] = str(p.resolve())

    if val_steps and val_ce:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(val_steps, val_ce, marker="o", markersize=2.5,
                color=colors["val"], linewidth=1.3, label="validation CE")
        if probe and any(v == v for v in probe):
            ax.plot(val_steps, probe, color=colors["probe"],
                    linewidth=1.0, label="training probe CE")
        if best_step is not None:
            try:
                best_i = val_steps.index(int(best_step))
                ax.scatter([val_steps[best_i]], [val_ce[best_i]],
                           color=colors["best"], zorder=5,
                           label="best val CE")
            except (ValueError, IndexError):
                pass
        ax.set_title("Validation CE")
        ax.set_xlabel("step")
        ax.set_ylabel("CE")
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=8)
        mark_events(ax)
        p = out_dir / "val_ce.png"
        _save(fig, p)
        paths["val_ce"] = str(p.resolve())
    else:
        p = out_dir / "val_ce.png"
        _empty_plot(plt, p, "Validation CE", "No validation history")
        paths["val_ce"] = str(p.resolve())

    # A combined image is convenient for reports and chat attachments.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    ax = axes[0, 0]
    if train_steps and lr:
        ax.plot(train_steps, lr, color=colors["lr"], linewidth=1.2)
    ax.set_title("Learning rate")
    ax.grid(alpha=0.15)
    mark_events(ax)
    ax = axes[0, 1]
    if train_steps and loss:
        ax.plot(train_steps, loss, color=colors["loss"], alpha=0.40,
                linewidth=0.7)
        if ema and any(v == v for v in ema):
            ax.plot(train_steps, ema, color=colors["ema"], linewidth=1.3)
    ax.set_title("Training loss / EMA")
    ax.grid(alpha=0.15)
    mark_events(ax)
    ax = axes[1, 0]
    if train_steps and grad and any(v == v for v in grad):
        ax.plot(train_steps, grad, color=colors["grad"], linewidth=0.9)
    else:
        ax.text(0.5, 0.5, "no grad-norm history", ha="center",
                va="center", transform=ax.transAxes, color="#666666",
                fontsize=9)
    ax.set_title("Gradient norm")
    ax.grid(alpha=0.15)
    mark_events(ax)
    ax = axes[1, 1]
    if val_steps and val_ce:
        ax.plot(val_steps, val_ce, marker="o", markersize=2,
                color=colors["val"], linewidth=1.1)
    ax.set_title("Validation CE")
    ax.grid(alpha=0.15)
    mark_events(ax)
    for ax in axes.flat:
        ax.set_xlabel("step")
    p = out_dir / "training_overview.png"
    _save(fig, p)
    paths["overview"] = str(p.resolve())
    return paths
