from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot predicted vs actual full328 validation-loss curves for one trained meta-model split."
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="meta-model-train/scripts/run_full328_timefit_residual.py",
        help="Path to the full328 training script used to define the model and data utilities.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_experiments/server_v1_full328_timefit_counts_stepmean_residual_4x/summary.json",
        help="Summary JSON used to pick the representative split by default.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_experiments/server_v1_full328_timefit_counts_stepmean_residual_4x/pred_curves",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Repeat index of the split to visualize. Defaults to the representative split from summary.json.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index of the split to visualize. Defaults to the representative split from summary.json.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=0.08,
        help="Alpha for individual architecture curves.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("auto", "cpu", "cuda"),
        help="Device used to refit the selected split.",
    )
    return parser


def load_train_module(train_script: Path):
    spec = importlib.util.spec_from_file_location("run_full328_timefit_residual", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summary(summary_json: Path) -> dict[str, object]:
    return json.loads(summary_json.read_text(encoding="utf-8"))


def choose_split(args: argparse.Namespace, summary: dict[str, object]) -> tuple[int, int]:
    if args.repeat is not None and args.fold is not None:
        return int(args.repeat), int(args.fold)
    rep = summary["representative_split"]
    return int(rep["repeat"]), int(rep["fold"])


def summarize_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(matrix, axis=0),
        "q10": np.nanpercentile(matrix, 10, axis=0),
        "q25": np.nanpercentile(matrix, 25, axis=0),
        "q75": np.nanpercentile(matrix, 75, axis=0),
        "q90": np.nanpercentile(matrix, 90, axis=0),
    }


def fit_linear_relation(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    y_line = slope * x_fit + intercept
    ss_res = float(np.sum((y_fit - y_line) ** 2))
    ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x_fit, y_fit)[0, 1]) if x_fit.size >= 2 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r,
        "r2": r2,
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


def build_curve_matrix(
    arch_ids: np.ndarray,
    step_indices: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    subset_arch_ids = sorted(set(int(x) for x in arch_ids[mask]))
    row_map = {arch_id: row_idx for row_idx, arch_id in enumerate(subset_arch_ids)}
    matrix = np.full((len(subset_arch_ids), num_steps), np.nan, dtype=np.float64)
    subset_indices = np.where(mask)[0]
    for idx in subset_indices:
        matrix[row_map[int(arch_ids[idx])], int(step_indices[idx])] = float(values[idx])
    return matrix


def fit_best_model_for_split(module, dataset, split: dict[str, object], summary: dict[str, object], device: torch.device):
    train_mask, val_mask, test_mask = module.build_split_masks(dataset, split)

    baseline_all = np.zeros(len(dataset.target), dtype=np.float32)
    for step_idx in range(len(dataset.step_values)):
        mask = train_mask & (dataset.step_indices == step_idx)
        baseline_all[dataset.step_indices == step_idx] = float(dataset.target[mask].mean())

    residual_target = dataset.target - baseline_all
    target_mean = float(residual_target[train_mask].mean())
    target_std = float(residual_target[train_mask].std())
    if target_std < 1e-8:
        target_std = 1.0

    count_features = torch.from_numpy(dataset.count_features).to(device=device, dtype=torch.float32)
    seq_tokens = torch.from_numpy(dataset.seq_tokens).to(device=device, dtype=torch.long)
    step_indices = torch.from_numpy(dataset.step_indices).to(device=device, dtype=torch.long)
    step_features = torch.from_numpy(dataset.step_features).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy((residual_target - target_mean) / target_std).to(device=device, dtype=torch.float32)
    train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
    val_idx = torch.from_numpy(np.where(val_mask)[0]).to(device)
    test_idx = torch.from_numpy(np.where(test_mask)[0]).to(device)

    seed = int(summary.get("torch_seed", 123)) + 100 * int(split["repeat"]) + int(split["fold"])
    module.set_seed(seed)
    model = module.TimefitMetaMLP(
        arch_mode=str(summary["arch_mode"]),
        time_mode=str(summary["time_mode"]),
        num_steps=len(dataset.step_values),
        embedding_dim=int(summary["embedding_dim"]),
        step_emb_dim=int(summary["step_emb_dim"]),
        hidden_dim=int(summary["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(summary["lr"]),
        weight_decay=float(summary["weight_decay"]),
    )

    max_epochs = int(summary["max_epochs"])
    patience = int(summary["patience"])
    min_epochs = int(summary.get("min_epochs", 0))
    best_epoch = -1
    best_selection = float("inf")
    epochs_since_best = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(count_features, seq_tokens, step_indices, step_features)
        loss = F.mse_loss(pred[train_idx], target_tensor[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_norm = model(count_features, seq_tokens, step_indices, step_features)
            val_objective_mse = float(F.mse_loss(pred_norm[val_idx], target_tensor[val_idx]).item())

        if val_objective_mse < best_selection - 1e-9:
            best_selection = val_objective_mse
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_since_best += 1

        if epoch + 1 >= min_epochs and epochs_since_best >= patience:
            break

    if best_state is None:
        raise RuntimeError("best state was never recorded")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm = model(count_features, seq_tokens, step_indices, step_features)
        pred_all = baseline_all + pred_norm.cpu().numpy() * target_std + target_mean

    metrics = {
        "train_mae": module.compute_mae(dataset.target[train_mask], pred_all[train_mask]),
        "val_mae": module.compute_mae(dataset.target[val_mask], pred_all[val_mask]),
        "test_mae": module.compute_mae(dataset.target[test_mask], pred_all[test_mask]),
        "train_objective_mse": float(F.mse_loss(pred_norm[train_idx], target_tensor[train_idx]).item()),
        "val_objective_mse": float(F.mse_loss(pred_norm[val_idx], target_tensor[val_idx]).item()),
        "test_objective_mse": float(F.mse_loss(pred_norm[test_idx], target_tensor[test_idx]).item()),
    }
    return {
        "best_epoch": best_epoch,
        "pred_all": pred_all,
        "actual_all": dataset.target.copy(),
        "baseline_all": baseline_all,
        "pred_norm_all": pred_norm.cpu().numpy(),
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "metrics": metrics,
    }


def style_axis(
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str | None,
    x_values: np.ndarray,
    log_x: bool,
    log_y: bool,
) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("training step")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(x_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(alpha=0.25)


def plot_predicted_vs_actual(
    output_png: Path,
    *,
    steps: np.ndarray,
    predicted_by_split: dict[str, np.ndarray],
    actual_by_split: dict[str, np.ndarray],
    split_sizes: dict[str, int],
    repeat: int,
    fold: int,
    best_epoch: int,
    line_alpha: float,
    min_step: int = 0,
    log_x: bool = False,
    log_y: bool = True,
) -> None:
    step_mask = steps >= min_step
    steps_plot = steps[step_mask]
    colors = {"train": "#1f77b4", "val": "#2a9d8f", "test": "#e76f51"}
    row_names = {"train": "Train subset", "val": "Validation subset", "test": "Test subset"}

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 12.0), sharex=False, sharey=False)
    for row_idx, split_name in enumerate(("train", "val", "test")):
        predicted = predicted_by_split[split_name][:, step_mask]
        actual = actual_by_split[split_name][:, step_mask]
        panels = [
            (axes[row_idx, 0], predicted, "Predicted curves"),
            (axes[row_idx, 1], actual, "Actual curves"),
        ]

        for ax, matrix, panel_name in panels:
            color = colors[split_name]
            stats = summarize_matrix(matrix)
            for curve in matrix:
                ax.plot(steps_plot, curve, color=color, alpha=line_alpha, linewidth=0.85)
            ax.fill_between(steps_plot, stats["q10"], stats["q90"], color=color, alpha=0.10, label="10-90%")
            ax.fill_between(steps_plot, stats["q25"], stats["q75"], color=color, alpha=0.18, label="25-75%")
            ax.plot(steps_plot, stats["median"], color=color, linewidth=2.2, label="median")
            style_axis(
                ax,
                title=f"{panel_name} ({row_names[split_name]}, n={split_sizes[split_name]})",
                ylabel="validation loss" if panel_name == "Predicted curves" else None,
                x_values=steps_plot,
                log_x=log_x,
                log_y=log_y,
            )
            if row_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")

        row_values = np.concatenate([predicted.reshape(-1), actual.reshape(-1)])
        row_values = row_values[np.isfinite(row_values) & (row_values > 0)]
        if row_values.size > 0:
            y_min = float(np.percentile(row_values, 1))
            y_max = float(np.percentile(row_values, 99.5))
            if y_min > 0 and y_max > y_min:
                axes[row_idx, 0].set_ylim(y_min * 0.9, y_max * 1.1)
                axes[row_idx, 1].set_ylim(y_min * 0.9, y_max * 1.1)

    step_tag = f"step >= {min_step}" if min_step > 0 else "all steps"
    axis_tag = "log-log" if log_x and log_y else ("log-y" if log_y else "linear")
    fig.suptitle(
        f"Full328 meta-model prediction vs ground truth, split r{repeat}f{fold}, selected epoch {best_epoch}\n"
        f"Target = validation-loss curves, {step_tag}, {axis_tag}",
        fontsize=15,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_step_scatter(
    output_png: Path,
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    repeat: int,
    fold: int,
    best_epoch: int,
    final_step: int,
) -> dict[str, float]:
    fit = fit_linear_relation(x_all, y_all)
    x_line = np.linspace(float(np.min(fit["x_fit"])), float(np.max(fit["x_fit"])), 200)
    y_line = float(fit["slope"]) * x_line + float(fit["intercept"])

    colors = {"train": "#1f77b4", "val": "#2a9d8f", "test": "#e76f51"}
    masks = {"train": train_mask, "val": val_mask, "test": test_mask}

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for split_name in ("train", "val", "test"):
        mask = masks[split_name]
        ax.scatter(
            x_all[mask],
            y_all[mask],
            s=26,
            alpha=0.70,
            color=colors[split_name],
            edgecolors="none",
            label=f"{split_name} subset",
        )

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=2.0,
        color="#444444",
        label="linear fit",
    )
    ax.set_xlabel(f"Actual architecture train loss at step {final_step}")
    ax.set_ylabel(f"Meta-model predicted loss at step {final_step}")
    ax.set_title(
        f"Final-step train loss vs meta-model prediction, split r{repeat}f{fold}, epoch {best_epoch}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")

    text = (
        f"Pearson r = {float(fit['r']):.4f}\n"
        f"$R^2$ = {float(fit['r2']):.4f}\n"
        f"y = {float(fit['slope']):.3f}x + {float(fit['intercept']):.4f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "slope": float(fit["slope"]),
        "intercept": float(fit["intercept"]),
        "pearson_r": float(fit["r"]),
        "r2": float(fit["r2"]),
    }


def main() -> None:
    args = build_argparser().parse_args()
    train_script = Path(args.train_script).resolve()
    summary_json = Path(args.summary_json).resolve()
    output_dir = Path(args.output_dir).resolve()

    module = load_train_module(train_script)
    summary = load_summary(summary_json)
    repeat, fold = choose_split(args, summary)
    device = module.pick_device(args.device)
    dataset = module.read_meta_rows(Path(summary["meta_csv"]), str(summary["target_name"]))
    train_dataset = module.read_meta_rows(Path(summary["meta_csv"]), "train")
    splits = module.build_grouped_splits(dataset.arch_ids.tolist(), int(summary["num_folds"]), int(summary["num_repeats"]))
    split = next(s for s in splits if int(s["repeat"]) == repeat and int(s["fold"]) == fold)

    fit = fit_best_model_for_split(module, dataset, split, summary, device)
    predicted_by_split = {}
    actual_by_split = {}
    split_sizes = {}
    for split_name, mask in (
        ("train", fit["train_mask"]),
        ("val", fit["val_mask"]),
        ("test", fit["test_mask"]),
    ):
        predicted_by_split[split_name] = build_curve_matrix(
            dataset.arch_ids,
            dataset.step_indices,
            fit["pred_all"],
            mask,
            num_steps=len(dataset.step_values),
        )
        actual_by_split[split_name] = build_curve_matrix(
            dataset.arch_ids,
            dataset.step_indices,
            fit["actual_all"],
            mask,
            num_steps=len(dataset.step_values),
        )
        split_sizes[split_name] = int(predicted_by_split[split_name].shape[0])

    stem = f"r{repeat}f{fold}_pred_vs_actual"
    plot_predicted_vs_actual(
        output_dir / f"{stem}.png",
        steps=dataset.step_values,
        predicted_by_split=predicted_by_split,
        actual_by_split=actual_by_split,
        split_sizes=split_sizes,
        repeat=repeat,
        fold=fold,
        best_epoch=int(fit["best_epoch"]),
        line_alpha=float(args.line_alpha),
        min_step=0,
        log_x=False,
        log_y=True,
    )
    plot_predicted_vs_actual(
        output_dir / f"{stem}_after_200_loglog.png",
        steps=dataset.step_values,
        predicted_by_split=predicted_by_split,
        actual_by_split=actual_by_split,
        split_sizes=split_sizes,
        repeat=repeat,
        fold=fold,
        best_epoch=int(fit["best_epoch"]),
        line_alpha=float(args.line_alpha),
        min_step=200,
        log_x=True,
        log_y=True,
    )

    final_step = int(dataset.step_values[-1])
    final_mask = dataset.step_values[dataset.step_indices] == final_step
    scatter_stats = plot_final_step_scatter(
        output_dir / f"{stem}_final_step_trainloss_scatter.png",
        x_all=train_dataset.target[final_mask],
        y_all=fit["pred_all"][final_mask],
        train_mask=fit["train_mask"][final_mask],
        val_mask=fit["val_mask"][final_mask],
        test_mask=fit["test_mask"][final_mask],
        repeat=repeat,
        fold=fold,
        best_epoch=int(fit["best_epoch"]),
        final_step=final_step,
    )

    result = {
        "repeat": repeat,
        "fold": fold,
        "best_epoch": int(fit["best_epoch"]),
        "metrics": fit["metrics"],
        "final_step_trainloss_vs_prediction": scatter_stats,
        "output_dir": str(output_dir),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
