from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import run_meta_baselines as baselines


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot final-step diagnostics for one meta-model configuration.")
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
    return p


def choose_split(
    splits: list[dict[str, object]],
    *,
    repeat: int | None,
    fold: int | None,
    cv_results_csv: str | None,
    config_id: str,
    representative_metric: str,
) -> tuple[dict[str, object], dict[str, float] | None]:
    if repeat is not None and fold is not None:
        split = next(s for s in splits if int(s["repeat"]) == repeat and int(s["fold"]) == fold)
        return split, None

    if cv_results_csv is None:
        raise SystemExit("either provide --repeat/--fold or pass --cv_results_csv to choose a representative split")

    cv_df = pd.read_csv(cv_results_csv)
    sub = cv_df[cv_df["config_id"] == config_id].copy()
    if sub.empty:
        raise SystemExit(f"config_id={config_id!r} not found in {cv_results_csv}")
    if representative_metric not in sub.columns:
        raise SystemExit(f"representative_metric={representative_metric!r} not found in {cv_results_csv}")

    target = float(sub[representative_metric].mean())
    sub["distance_to_mean"] = (sub[representative_metric] - target).abs()
    rep = sub.sort_values(["distance_to_mean", "repeat", "fold"]).iloc[0]
    split = next(s for s in splits if int(s["repeat"]) == int(rep["repeat"]) and int(s["fold"]) == int(rep["fold"]))
    return split, {
        "repeat": int(rep["repeat"]),
        "fold": int(rep["fold"]),
        "metric_value": float(rep[representative_metric]),
        "metric_mean": target,
    }


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks + 1.0


def fit_linear_relation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": pearson_r,
        "r2": r2,
    }


def refit_split(
    dataset: baselines.MetaDataset,
    split: dict[str, object],
    *,
    model_name: str,
    target_name: str,
    target_mode: str,
    target_transform: str,
    hidden_dim: int,
    embedding_dim: int,
    sequence_gru_head_dim: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    torch_seed: int,
    device: torch.device,
) -> dict[str, object]:
    baselines.set_seed(torch_seed + 97 * int(split["repeat"]) + 13 * int(split["fold"]))
    tensors = baselines.build_split_tensors(dataset, split, device)
    seq_tokens = tensors["seq_tokens"]
    count_features = tensors["count_features"]
    step_features = tensors["step_features"]
    train_mask = tensors["train_mask"]
    val_mask = tensors["val_mask"]
    test_mask = tensors["test_mask"]
    train_mask_np = train_mask.cpu().numpy()

    target_all_raw = baselines.select_target_array(dataset, target_name)
    target_all_model = baselines.transform_target_array(target_all_raw, target_transform)
    step_mean_pred_all = baselines.compute_step_mean_predictions(dataset, target_all_model, train_mask_np)

    if target_mode == "direct":
        baseline_all = np.zeros_like(target_all_model, dtype=np.float32)
    elif target_mode == "residual_over_step_mean":
        baseline_all = step_mean_pred_all
    else:
        raise ValueError(f"unknown target_mode={target_mode}")

    model_target_all = target_all_model - baseline_all
    target_norm, target_mean, target_std = baselines.standardize_target(model_target_all[train_mask_np], model_target_all)
    target_tensor = torch.from_numpy(target_norm).to(device=device, dtype=torch.float32)

    model = baselines.build_model(
        model_name,
        count_dim=dataset.count_features.shape[1],
        vocab_size=dataset.vocab_size,
        seq_len=dataset.seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        out_dim=1 if target_name != "joint" else 2,
        sequence_gru_head_dim=sequence_gru_head_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_since_best = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = baselines.model_forward(model, model_name, seq_tokens, count_features, step_features)
        train_loss = F.mse_loss(pred[train_mask], target_tensor[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = baselines.model_forward(model, model_name, seq_tokens, count_features, step_features)
            val_loss = float(F.mse_loss(pred[val_mask], target_tensor[val_mask]).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= patience:
            break

    if best_state is None:
        raise AssertionError("best_state was never set")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm = baselines.model_forward(model, model_name, seq_tokens, count_features, step_features).cpu().numpy()

    pred_all_model = pred_norm * target_std + target_mean + baseline_all
    pred_all_raw = baselines.inverse_target_array(pred_all_model, target_transform)
    return {
        "best_epoch": best_epoch,
        "train_mask": train_mask_np,
        "test_mask": test_mask.cpu().numpy(),
        "pred_all_raw": pred_all_raw,
        "target_all_raw": target_all_raw,
        "baseline_all": baseline_all,
    }


def build_test_frame(
    rows: list[dict[str, object]],
    dataset: baselines.MetaDataset,
    fit: dict[str, object],
    *,
    target_name: str,
) -> pd.DataFrame:
    if target_name != "val":
        raise SystemExit("this diagnostic script currently supports target_name=val only")

    final_step = int(dataset.steps.max())
    final_mask = np.logical_and(dataset.steps == final_step, fit["test_mask"])
    indices = np.where(final_mask)[0]

    true_loss = fit["target_all_raw"][:, 0]
    pred_loss = fit["pred_all_raw"][:, 0]
    baseline_log = fit["baseline_all"][:, 0]
    true_log = np.log(np.clip(true_loss, 1e-12, None))
    pred_log = np.log(np.clip(pred_loss, 1e-12, None))

    frame = pd.DataFrame(
        {
            "architecture_id": dataset.arch_ids[indices],
            "architecture_code": [str(rows[idx]["architecture_code"]) for idx in indices],
            "true_loss": true_loss[indices],
            "pred_loss": pred_loss[indices],
            "baseline_log_step_mean": baseline_log[indices],
            "true_log_loss": true_log[indices],
            "pred_log_loss": pred_log[indices],
            "true_log_residual": true_log[indices] - baseline_log[indices],
            "pred_log_residual": pred_log[indices] - baseline_log[indices],
        }
    )
    frame["true_rank"] = average_tie_ranks(frame["true_loss"].to_numpy(dtype=np.float64))
    frame["pred_rank"] = average_tie_ranks(frame["pred_loss"].to_numpy(dtype=np.float64))
    return frame


def plot_xy_scatter(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    stats: dict[str, float],
    identity: bool = True,
) -> None:
    ax.scatter(x, y, s=38, alpha=0.8, color="#2a9d8f", edgecolors="none")
    x_min = float(min(np.min(x), np.min(y)))
    x_max = float(max(np.max(x), np.max(y)))
    pad = 0.06 * (x_max - x_min) if x_max > x_min else 0.5
    lower = x_min - pad
    upper = x_max + pad
    if identity:
        ax.plot([lower, upper], [lower, upper], linestyle=":", color="#666666", linewidth=1.6, label="y = x")
    line_x = np.linspace(lower, upper, 200)
    ax.plot(
        line_x,
        stats["slope"] * line_x + stats["intercept"],
        linestyle="--",
        color="#e76f51",
        linewidth=2.0,
        label="linear fit",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=9)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = baselines.read_meta_rows(args.meta_csv)
    dataset = baselines.build_meta_dataset(rows)
    device = baselines.pick_device(args.device)
    splits = baselines.build_grouped_splits(dataset.arch_ids.tolist(), args.num_folds, args.num_repeats)
    split, rep_info = choose_split(
        splits,
        repeat=args.repeat,
        fold=args.fold,
        cv_results_csv=args.cv_results_csv,
        config_id=args.config_id,
        representative_metric=args.representative_metric,
    )

    fit = refit_split(
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
    frame = build_test_frame(rows, dataset, fit, target_name=args.target_name)

    log_stats = fit_linear_relation(frame["true_log_loss"].to_numpy(), frame["pred_log_loss"].to_numpy())
    residual_stats = fit_linear_relation(frame["true_log_residual"].to_numpy(), frame["pred_log_residual"].to_numpy())
    rank_stats = fit_linear_relation(frame["true_rank"].to_numpy(), frame["pred_rank"].to_numpy())

    log_spearman = baselines.compute_spearman(frame["true_log_loss"].to_numpy(), frame["pred_log_loss"].to_numpy())
    rank_spearman = baselines.compute_spearman(frame["true_loss"].to_numpy(), frame["pred_loss"].to_numpy())

    frame_path = out_dir / f"{args.config_id}_representative_split_test_final_step.csv"
    frame.to_csv(frame_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8))
    plot_xy_scatter(
        axes[0],
        x=frame["true_log_loss"].to_numpy(),
        y=frame["pred_log_loss"].to_numpy(),
        title="Final-step log loss",
        xlabel="True log(val loss)",
        ylabel="Predicted log(val loss)",
        stats=log_stats,
    )
    plot_xy_scatter(
        axes[1],
        x=frame["true_log_residual"].to_numpy(),
        y=frame["pred_log_residual"].to_numpy(),
        title="Final-step log-loss residual",
        xlabel="True residual over step_mean(log loss)",
        ylabel="Pred residual over step_mean(log loss)",
        stats=residual_stats,
    )
    plot_xy_scatter(
        axes[2],
        x=frame["true_rank"].to_numpy(),
        y=frame["pred_rank"].to_numpy(),
        title="Architecture ranking",
        xlabel="True rank (1 = best)",
        ylabel="Predicted rank (1 = best)",
        stats=rank_stats,
    )

    for ax, text_lines in (
        (
            axes[0],
            [
                f"Spearman = {log_spearman:.4f}",
                f"slope = {log_stats['slope']:.4f}",
                f"intercept = {log_stats['intercept']:.4f}",
                f"R^2 = {log_stats['r2']:.4f}",
            ],
        ),
        (
            axes[1],
            [
                f"slope = {residual_stats['slope']:.4f}",
                f"intercept = {residual_stats['intercept']:.4f}",
                f"R^2 = {residual_stats['r2']:.4f}",
                f"MAE = {float(np.mean(np.abs(frame['pred_log_residual'] - frame['true_log_residual']))):.4f}",
            ],
        ),
        (
            axes[2],
            [
                f"Spearman = {rank_spearman:.4f}",
                f"slope = {rank_stats['slope']:.4f}",
                f"intercept = {rank_stats['intercept']:.4f}",
                f"R^2 = {rank_stats['r2']:.4f}",
            ],
        ),
    ):
        ax.text(
            0.03,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )

    final_step = int(dataset.steps.max())
    rep_text = ""
    if rep_info is not None:
        rep_text = (
            f", representative split r{rep_info['repeat']}f{rep_info['fold']} "
            f"({args.representative_metric}={rep_info['metric_value']:.4f}, mean={rep_info['metric_mean']:.4f})"
        )
    fig.suptitle(
        f"{args.config_id}: final-step test diagnostics at recorded step {final_step} (1000th training step){rep_text}",
        fontsize=15,
    )
    fig.tight_layout()

    png_path = out_dir / f"{args.config_id}_representative_split_test_final_step_diagnostics.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "config_id": args.config_id,
        "repeat": int(split["repeat"]),
        "fold": int(split["fold"]),
        "final_recorded_step": final_step,
        "interpreted_training_step": final_step + 1,
        "num_test_architectures": int(len(frame)),
        "representative_metric": args.representative_metric,
        "representative_info": rep_info,
        "log_loss_scatter": {**log_stats, "spearman": float(log_spearman)},
        "log_residual_scatter": {
            **residual_stats,
            "mae": float(np.mean(np.abs(frame["pred_log_residual"] - frame["true_log_residual"]))),
        },
        "rank_scatter": {**rank_stats, "spearman": float(rank_spearman)},
        "output_png": str(png_path),
        "output_csv": str(frame_path),
    }
    json_path = out_dir / f"{args.config_id}_representative_split_test_final_step_diagnostics.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
