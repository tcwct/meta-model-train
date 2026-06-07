from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot representative train/val curves from a trained architecture family.")
    p.add_argument("--family_dir", type=str, required=True)
    p.add_argument("--num_best", type=int, default=3)
    p.add_argument("--num_worst", type=int, default=3)
    p.add_argument("--output_png", type=str, default=None)
    p.add_argument("--output_late_png", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    return p


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"family summary is empty: {path}")
    return rows


def choose_rows(rows: list[dict[str, str]], num_best: int, num_worst: int) -> list[dict[str, str]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["final_val_loss"]))
    chosen: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in sorted_rows[:num_best]:
        code = row["architecture_code"]
        if code not in seen:
            chosen.append(row)
            seen.add(code)
    for row in reversed(sorted_rows[-num_worst:]):
        code = row["architecture_code"]
        if code not in seen:
            chosen.append(row)
            seen.add(code)
    return chosen


def load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_curve_arrays(metrics: list[dict[str, str]]) -> dict[str, list[float]]:
    return {
        "steps": [int(m["step"]) for m in metrics],
        "train_steps": [int(m["step"]) for m in metrics if m["train_loss"]],
        "train_loss": [float(m["train_loss"]) for m in metrics if m["train_loss"]],
        "val_loss": [float(m["val_loss"]) for m in metrics if m["val_loss"]],
    }


def plot_selected_curves(chosen_rows: list[dict[str, str]], output_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, row in zip(axes, chosen_rows):
        run_dir = Path(row["run_dir"])
        metrics = load_metrics(run_dir / "metrics.csv")
        arrays = _extract_curve_arrays(metrics)

        ax.plot(arrays["train_steps"], arrays["train_loss"], linestyle="--", linewidth=1.8, label="train")
        ax.plot(arrays["steps"], arrays["val_loss"], linestyle="-", linewidth=2.2, label="val")
        ax.set_title(
            f"{row['architecture_code']}\nfinal val={float(row['final_val_loss']):.4f}",
            fontsize=10,
        )
        ax.grid(alpha=0.25)

    for ax in axes[len(chosen_rows):]:
        ax.axis("off")

    axes[0].legend(frameon=False, loc="upper right")
    fig.supxlabel("training step")
    fig.supylabel("MSE loss")
    fig.suptitle("Representative minimal-architecture curves", fontsize=14)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_late_stage_curves(chosen_rows: list[dict[str, str]], output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    late_vals: list[float] = []

    for row in chosen_rows:
        run_dir = Path(row["run_dir"])
        arrays = _extract_curve_arrays(load_metrics(run_dir / "metrics.csv"))
        late_points = [(step, val) for step, val in zip(arrays["steps"], arrays["val_loss"]) if step >= 100]
        late_steps = [step for step, _ in late_points]
        late_val = [val for _, val in late_points]
        late_vals.extend(late_val)
        ax.plot(late_steps, late_val, linewidth=2.0, label=row["architecture_code"])

    if late_vals:
        y_min = min(late_vals)
        y_max = max(late_vals)
        pad = max(0.0005, 0.1 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title("Late-stage validation curves (step >= 100)")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    family_dir = Path(args.family_dir).resolve()
    summary_path = family_dir / "family_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"missing family summary: {summary_path}")

    rows = load_summary_rows(summary_path)
    chosen_rows = choose_rows(rows, num_best=args.num_best, num_worst=args.num_worst)
    if not chosen_rows:
        raise SystemExit("no rows selected for plotting")

    output_png = Path(args.output_png).resolve() if args.output_png else family_dir / "representative_curves.png"
    output_late_png = (
        Path(args.output_late_png).resolve() if args.output_late_png else family_dir / "representative_curves_late.png"
    )
    output_json = Path(args.output_json).resolve() if args.output_json else family_dir / "representative_curves_selection.json"

    plot_selected_curves(chosen_rows, output_png)
    plot_late_stage_curves(chosen_rows, output_late_png)
    output_json.write_text(json.dumps(chosen_rows, indent=2), encoding="utf-8")

    print(f"[plot_family_curves] wrote_png={output_png}")
    print(f"[plot_family_curves] wrote_late_png={output_late_png}")
    print(f"[plot_family_curves] wrote_selection={output_json}")


if __name__ == "__main__":
    main()

