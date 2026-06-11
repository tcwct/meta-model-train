from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_meta_baselines as baselines
import plot_meta_model_final_step_diagnostics as diag


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot test-set log-residual scatter at multiple steps for one meta-model config.")
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--config_id", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--target_name", type=str, default="val", choices=("train", "val", "joint"))
    p.add_argument("--target_mode", type=str, default="residual_over_step_mean", choices=("direct", "residual_over_step_mean"))
    p.add_argument("--target_transform", type=str, default="log", choices=("raw", "log"))
    p.add_argument("--hidden_dim", type=int, required=True)
    p.add_argument("--embedding_dim", type=int, default=1)
    p.add_argument("--sequence_gru_head_dim", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_folds", type=int, default=3)
    p.add_argument("--num_repeats", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--torch_seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda"))
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--cv_results_csv", type=str, default=None)
    p.add_argument("--representative_metric", type=str, default="test_spearman_step_999")
    p.add_argument("--steps", type=str, default="0,50,100,300,600,999")
    return p


def parse_steps(text: str) -> list[int]:
    values = [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise SystemExit("expected at least one step")
    return values


def build_step_frame(
    dataset: baselines.MetaDataset,
    fit: dict[str, object],
    *,
    step: int,
    target_name: str,
) -> pd.DataFrame:
    if target_name != "val":
        raise SystemExit("this diagnostic currently supports target_name=val only")

    mask = np.logical_and(dataset.steps == step, fit["test_mask"])
    idx = np.where(mask)[0]
    true_loss = fit["target_all_raw"][:, 0]
    pred_loss = fit["pred_all_raw"][:, 0]
    baseline_log = fit["baseline_all"][:, 0]
    true_log = np.log(np.clip(true_loss, 1e-12, None))
    pred_log = np.log(np.clip(pred_loss, 1e-12, None))

    frame = pd.DataFrame(
        {
            "architecture_id": dataset.arch_ids[idx],
            "step": step,
            "true_log_residual": true_log[idx] - baseline_log[idx],
            "pred_log_residual": pred_log[idx] - baseline_log[idx],
            "true_log_loss": true_log[idx],
            "pred_log_loss": pred_log[idx],
        }
    )
    return frame


def plot_step_panel(ax: plt.Axes, frame: pd.DataFrame, step: int, stats: dict[str, float], spearman: float, mae: float) -> None:
    x = frame["true_log_residual"].to_numpy(dtype=float)
    y = frame["pred_log_residual"].to_numpy(dtype=float)
    ax.scatter(x, y, s=36, alpha=0.8, color="#2a9d8f", edgecolors="none")

    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    pad = 0.08 * (high - low) if high > low else 0.5
    lower = low - pad
    upper = high + pad
    ax.plot([lower, upper], [lower, upper], linestyle=":", color="#666666", linewidth=1.5, label="y = x")
    line_x = np.linspace(lower, upper, 200)
    ax.plot(line_x, stats["slope"] * line_x + stats["intercept"], linestyle="--", color="#e76f51", linewidth=2.0, label="linear fit")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(f"step {step}", fontsize=12)
    ax.set_xlabel("True log residual")
    ax.set_ylabel("Pred log residual")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"Spearman = {spearman:.3f}",
                f"slope = {stats['slope']:.3f}",
                f"intercept = {stats['intercept']:.3f}",
                f"R^2 = {stats['r2']:.3f}",
                f"MAE = {mae:.3f}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = parse_steps(args.steps)
    rows = baselines.read_meta_rows(args.meta_csv)
    dataset = baselines.build_meta_dataset(rows)
    device = baselines.pick_device(args.device)
    splits = baselines.build_grouped_splits(dataset.arch_ids.tolist(), args.num_folds, args.num_repeats)
    split, rep_info = diag.choose_split(
        splits,
        repeat=args.repeat,
        fold=args.fold,
        cv_results_csv=args.cv_results_csv,
        config_id=args.config_id,
        representative_metric=args.representative_metric,
    )

    fit = diag.refit_split(
        dataset,
        split,
        model_name=args.model_name,
        target_name=args.target_name,
        target_mode=args.target_mode,
        target_transform=args.target_transform,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        sequence_gru_head_dim=args.sequence_gru_head_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        torch_seed=args.torch_seed,
        device=device,
    )

    frames: list[pd.DataFrame] = []
    summary_steps: list[dict[str, float | int]] = []
    ncols = 3
    nrows = int(np.ceil(len(steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.5, 5.6 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, step in zip(axes_arr.flatten(), steps):
        frame = build_step_frame(dataset, fit, step=step, target_name=args.target_name)
        frames.append(frame)
        x = frame["true_log_residual"].to_numpy(dtype=float)
        y = frame["pred_log_residual"].to_numpy(dtype=float)
        stats = diag.fit_linear_relation(x, y)
        spearman = baselines.compute_spearman(x, y)
        mae = float(np.mean(np.abs(y - x)))
        summary_steps.append(
            {
                "step": step,
                "num_test_architectures": int(len(frame)),
                "spearman": float(spearman),
                "slope": float(stats["slope"]),
                "intercept": float(stats["intercept"]),
                "pearson_r": float(stats["pearson_r"]),
                "r2": float(stats["r2"]),
                "mae": mae,
            }
        )
        plot_step_panel(ax, frame, step, stats, spearman, mae)

    for ax in axes_arr.flatten()[len(steps):]:
        ax.axis("off")

    rep_text = ""
    if rep_info is not None:
        rep_text = (
            f", representative split r{rep_info['repeat']}f{rep_info['fold']} "
            f"({args.representative_metric}={rep_info['metric_value']:.4f}, mean={rep_info['metric_mean']:.4f})"
        )
    fig.suptitle(
        f"{args.config_id}: test-set log-residual scatter across steps{rep_text}",
        fontsize=16,
    )
    fig.tight_layout()

    stem = f"{args.config_id}_representative_split_log_residual_by_step"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    frame_all = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / f"{stem}.csv"
    frame_all.to_csv(csv_path, index=False)

    summary = {
        "config_id": args.config_id,
        "repeat": int(split["repeat"]),
        "fold": int(split["fold"]),
        "steps": steps,
        "representative_metric": args.representative_metric,
        "representative_info": rep_info,
        "per_step": summary_steps,
        "output_png": str(png_path),
        "output_csv": str(csv_path),
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
