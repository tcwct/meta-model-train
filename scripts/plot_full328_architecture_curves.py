from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot aggregate visualizations for the 328 architecture train/val loss curves."
    )
    parser.add_argument(
        "--meta_csv",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_family/server_v1_full328_meta_dataset.csv",
        help="Path to the full328 meta dataset CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_family/full328_curve_visualizations",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=0.06,
        help="Alpha for individual curves in the spaghetti plot.",
    )
    return parser


def load_curve_matrix(meta_csv: Path, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    curves_by_arch: dict[str, dict[int, float]] = defaultdict(dict)
    steps_seen: set[int] = set()
    with meta_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if step < 0:
                continue
            value_str = row[metric_key].strip()
            if not value_str:
                continue
            curves_by_arch[row["architecture_id"]][step] = float(value_str)
            steps_seen.add(step)

    steps = np.asarray(sorted(steps_seen), dtype=np.int32)
    matrix = np.full((len(curves_by_arch), len(steps)), np.nan, dtype=np.float64)
    step_to_col = {step: idx for idx, step in enumerate(steps)}
    for row_idx, arch_id in enumerate(sorted(curves_by_arch, key=lambda x: int(x))):
        for step, value in curves_by_arch[arch_id].items():
            matrix[row_idx, step_to_col[step]] = value
    return steps, matrix


def summarize_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(matrix, axis=0),
        "q10": np.nanpercentile(matrix, 10, axis=0),
        "q25": np.nanpercentile(matrix, 25, axis=0),
        "q75": np.nanpercentile(matrix, 75, axis=0),
        "q90": np.nanpercentile(matrix, 90, axis=0),
    }


def style_axis(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title)
    ax.set_xlabel("training step")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)


def plot_spaghetti(
    output_png: Path,
    steps: np.ndarray,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    line_alpha: float,
    min_step: int = 0,
) -> None:
    train_stats = summarize_matrix(train_matrix)
    val_stats = summarize_matrix(val_matrix)
    colors = {"train": "#1f77b4", "val": "#2a9d8f"}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    panels = [
        (axes[0], "Train loss across 328 architectures", "loss", "train", train_matrix, train_stats),
        (axes[1], "Validation loss across 328 architectures", None, "val", val_matrix, val_stats),
    ]
    for ax, title, ylabel, split_name, matrix, stats in panels:
        color = colors[split_name]
        for curve in matrix:
            ax.plot(steps, curve, color=color, alpha=line_alpha, linewidth=0.8)
        ax.fill_between(steps, stats["q10"], stats["q90"], color=color, alpha=0.12, label="10-90% band")
        ax.fill_between(steps, stats["q25"], stats["q75"], color=color, alpha=0.22, label="25-75% band")
        ax.plot(steps, stats["median"], color=color, linewidth=2.4, label="median")
        style_axis(ax, title, ylabel)
        ax.set_xlim(left=min_step)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    if min_step > 0:
        fig.suptitle(f"Full328 architecture curves: spaghetti view (step >= {min_step})", fontsize=18)
    else:
        fig.suptitle("Full328 architecture curves: spaghetti view", fontsize=18)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_quantiles(
    output_png: Path,
    steps: np.ndarray,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    min_step: int = 0,
) -> None:
    train_stats = summarize_matrix(train_matrix)
    val_stats = summarize_matrix(val_matrix)
    colors = {"train": "#1f77b4", "val": "#2a9d8f"}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    panels = [
        (axes[0], "Train loss quantile bands", "loss", "train", train_stats),
        (axes[1], "Validation loss quantile bands", None, "val", val_stats),
    ]
    for ax, title, ylabel, split_name, stats in panels:
        color = colors[split_name]
        ax.fill_between(steps, stats["q10"], stats["q90"], color=color, alpha=0.12, label="10-90% band")
        ax.fill_between(steps, stats["q25"], stats["q75"], color=color, alpha=0.24, label="25-75% band")
        ax.plot(steps, stats["median"], color=color, linewidth=2.6, label="median")
        style_axis(ax, title, ylabel)
        ax.set_xlim(left=min_step)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    if min_step > 0:
        fig.suptitle(f"Full328 architecture curves: quantile-band view (step >= {min_step})", fontsize=18)
    else:
        fig.suptitle("Full328 architecture curves: quantile-band view", fontsize=18)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    meta_csv = Path(args.meta_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    steps_train, train_matrix = load_curve_matrix(meta_csv, "train_loss")
    steps_val, val_matrix = load_curve_matrix(meta_csv, "val_loss")
    if not np.array_equal(steps_train, steps_val):
        raise ValueError("Train and validation steps are not aligned.")

    plot_spaghetti(
        output_dir / "full328_architecture_curves_spaghetti.png",
        steps_train,
        train_matrix,
        val_matrix,
        line_alpha=float(args.line_alpha),
    )
    plot_quantiles(
        output_dir / "full328_architecture_curves_quantiles.png",
        steps_train,
        train_matrix,
        val_matrix,
    )
    plot_spaghetti(
        output_dir / "full328_architecture_curves_spaghetti_after_200.png",
        steps_train,
        train_matrix,
        val_matrix,
        line_alpha=float(args.line_alpha),
        min_step=200,
    )
    plot_quantiles(
        output_dir / "full328_architecture_curves_quantiles_after_200.png",
        steps_train,
        train_matrix,
        val_matrix,
        min_step=200,
    )
    print(f"[plot_full328_architecture_curves] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
