from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot step-wise loss spread across architectures for the meta dataset.")
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--output_png", type=str, required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    meta_csv = Path(args.meta_csv).resolve()
    output_png = Path(args.output_png).resolve()

    val_groups: dict[int, list[float]] = defaultdict(list)
    train_groups: dict[int, list[float]] = defaultdict(list)

    with meta_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if step < 0:
                continue
            train_groups[step].append(float(row["train_loss"]))
            val_groups[step].append(float(row["val_loss"]))

    steps = sorted(val_groups)
    val_mean = [statistics.mean(val_groups[s]) for s in steps]
    val_std = [statistics.pstdev(val_groups[s]) for s in steps]
    val_min = [min(val_groups[s]) for s in steps]
    val_max = [max(val_groups[s]) for s in steps]

    train_mean = [statistics.mean(train_groups[s]) for s in steps]
    train_std = [statistics.pstdev(train_groups[s]) for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    axes[0].plot(steps, val_mean, label="val mean", linewidth=2.2)
    axes[0].fill_between(steps, val_min, val_max, alpha=0.18, label="val min-max")
    axes[0].fill_between(
        steps,
        [m - s for m, s in zip(val_mean, val_std)],
        [m + s for m, s in zip(val_mean, val_std)],
        alpha=0.28,
        label="val mean ± std",
    )
    axes[0].set_title("Validation loss spread across architectures")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(steps, train_mean, label="train mean", linewidth=2.2)
    axes[1].fill_between(
        steps,
        [m - s for m, s in zip(train_mean, train_std)],
        [m + s for m, s in zip(train_mean, train_std)],
        alpha=0.28,
        label="train mean ± std",
    )
    axes[1].set_title("Train loss spread across architectures")
    axes[1].set_xlabel("training step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_meta_dataset_spread] wrote={output_png}")


if __name__ == "__main__":
    main()
