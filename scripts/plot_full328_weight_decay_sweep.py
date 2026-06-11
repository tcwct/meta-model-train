from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_GROUPS = {
    "mae": ["train_mae", "val_mae", "test_mae"],
    "objective_mse": ["train_objective_mse", "val_objective_mse", "test_objective_mse"],
}

COLORS = {
    "train_mae": "#1f77b4",
    "val_mae": "#2a9d8f",
    "test_mae": "#e76f51",
    "train_objective_mse": "#1f77b4",
    "val_objective_mse": "#2a9d8f",
    "test_objective_mse": "#e76f51",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate full328 weight-decay sweep results.")
    parser.add_argument(
        "--experiments_root",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_experiments",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_experiments/server_v1_full328_timefit_counts_stepmean_residual_weight_decay_sweep",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Experiment directories to aggregate.",
    )
    return parser


def ensure_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_experiment(experiment_dir: Path) -> dict[str, float | str]:
    summary = read_json(experiment_dir / "summary.json")
    weight_decay = float(summary["weight_decay"])
    row: dict[str, float | str] = {
        "experiment_dir": str(experiment_dir),
        "weight_decay": weight_decay,
        "selected_epoch_mean": float(summary["selected_epoch_mean"]),
        "selected_epoch_std": float(summary["selected_epoch_std"]),
        "representative_repeat": int(summary["representative_split"]["repeat"]),
        "representative_fold": int(summary["representative_split"]["fold"]),
        "representative_selected_epoch": int(summary["representative_split"]["selected_epoch"]),
    }
    for metric_group in ("selected_checkpoint_summary", "baseline_summary"):
        prefix = "selected" if metric_group == "selected_checkpoint_summary" else "baseline"
        metric_summary = summary[metric_group]
        for key, value in metric_summary.items():
            row[f"{prefix}_{key}"] = float(value)
    return row


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_tick_labels(values: list[float]) -> list[str]:
    labels = []
    for value in values:
        if value == 0.0:
            labels.append("0")
        else:
            labels.append(f"{value:.0e}")
    return labels


def plot_metric_group(
    ax: plt.Axes,
    rows: list[dict[str, float | str]],
    metric_keys: list[str],
    ylabel: str,
) -> None:
    x = np.arange(len(rows))
    for key in metric_keys:
        means = np.asarray([float(row[f"selected_{key}_mean"]) for row in rows], dtype=np.float64)
        stds = np.asarray([float(row[f"selected_{key}_std"]) for row in rows], dtype=np.float64)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.0,
            capsize=3,
            color=COLORS[key],
            label=key,
        )
        baseline_value = float(rows[0][f"baseline_{key}_mean"])
        ax.axhline(baseline_value, linestyle="--", linewidth=1.2, color=COLORS[key], alpha=0.8)

    ax.set_xticks(x, format_tick_labels([float(row["weight_decay"]) for row in rows]))
    ax.set_xlabel("weight decay")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def plot_selected_epoch(ax: plt.Axes, rows: list[dict[str, float | str]]) -> None:
    x = np.arange(len(rows))
    means = np.asarray([float(row["selected_epoch_mean"]) for row in rows], dtype=np.float64)
    stds = np.asarray([float(row["selected_epoch_std"]) for row in rows], dtype=np.float64)
    ax.errorbar(x, means, yerr=stds, marker="o", linewidth=2.0, capsize=3, color="#6d597a")
    ax.set_xticks(x, format_tick_labels([float(row["weight_decay"]) for row in rows]))
    ax.set_xlabel("weight decay")
    ax.set_ylabel("selected epoch")
    ax.set_title("Selection epoch across splits")
    ax.grid(alpha=0.25)


def main() -> None:
    args = build_argparser().parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    out_dir = ensure_dir(args.output_dir)

    rows = []
    for name in args.dirs:
        experiment_dir = (experiments_root / name).resolve()
        rows.append(summarize_experiment(experiment_dir))
    rows.sort(key=lambda row: float(row["weight_decay"]))

    write_csv(out_dir / "weight_decay_sweep_summary.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    plot_metric_group(axes[0], rows, METRIC_GROUPS["mae"], ylabel="MAE")
    axes[0].set_title("Selected-checkpoint MAE")
    plot_metric_group(axes[1], rows, METRIC_GROUPS["objective_mse"], ylabel="objective MSE")
    axes[1].set_title("Selected-checkpoint objective MSE")
    plot_selected_epoch(axes[2], rows)
    fig.suptitle("Full328 heldout counts_stepmean_residual: weight decay sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_decay_sweep_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
