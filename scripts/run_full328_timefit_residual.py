from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_TOKEN_TO_ID = {"L": 0, "A": 1, "R": 2}


@dataclass
class MetaDataset:
    arch_ids: np.ndarray
    seq_tokens: np.ndarray
    count_features: np.ndarray
    step_features: np.ndarray
    step_indices: np.ndarray
    step_values: np.ndarray
    target: np.ndarray


class TimefitMetaMLP(nn.Module):
    def __init__(
        self,
        *,
        arch_mode: str,
        time_mode: str,
        num_steps: int,
        embedding_dim: int,
        step_emb_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.arch_mode = arch_mode
        self.time_mode = time_mode

        if arch_mode == "counts":
            arch_dim = 3
            self.token_embedding = None
        elif arch_mode == "sequence_concat":
            arch_dim = 6 * embedding_dim
            self.token_embedding = nn.Embedding(3, embedding_dim)
        else:
            raise ValueError(f"unknown arch_mode={arch_mode}")

        if time_mode == "continuous":
            time_dim = 2
            self.step_embedding = None
        elif time_mode == "discrete":
            time_dim = step_emb_dim
            self.step_embedding = nn.Embedding(num_steps, step_emb_dim)
        else:
            raise ValueError(f"unknown time_mode={time_mode}")

        input_dim = arch_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        count_features: torch.Tensor,
        seq_tokens: torch.Tensor,
        step_indices: torch.Tensor,
        step_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.arch_mode == "counts":
            arch_repr = count_features
        else:
            arch_repr = self.token_embedding(seq_tokens).reshape(seq_tokens.shape[0], -1)

        if self.time_mode == "continuous":
            time_repr = step_features
        else:
            time_repr = self.step_embedding(step_indices)

        x = torch.cat((arch_repr, time_repr), dim=-1)
        return self.net(x).squeeze(-1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train heldout counts_stepmean_residual on the full 328-architecture meta dataset."
    )
    parser.add_argument(
        "--meta_csv",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_family/server_v1_full328_meta_dataset.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="meta-model-train/outputs/toy_diffusion/meta_model_experiments/server_v1_full328_timefit_counts_stepmean_residual_4x",
    )
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--target_name", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--num_folds", type=int, default=4)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=2)
    parser.add_argument("--step_emb_dim", type=int, default=2)
    parser.add_argument("--torch_seed", type=int, default=123)
    return parser


def ensure_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_architecture_code(code: str) -> list[int]:
    return [ARCH_TOKEN_TO_ID[token] for token in code.split("-")]


def encode_step_features(step: int, max_step: int) -> list[float]:
    return [step / max_step, math.log1p(step) / math.log1p(max_step)]


def read_meta_rows(meta_csv: Path, target_name: str) -> MetaDataset:
    rows: list[dict[str, str]] = []
    with meta_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            if step < 0:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no usable rows found in {meta_csv}")

    max_step = max(int(row["step"]) for row in rows)
    step_values = sorted({int(row["step"]) for row in rows})
    step_to_idx = {step: idx for idx, step in enumerate(step_values)}
    arch_ids = np.asarray([int(row["architecture_id"]) for row in rows], dtype=np.int64)
    seq_tokens = np.asarray([parse_architecture_code(row["architecture_code"]) for row in rows], dtype=np.int64)
    count_features = np.asarray(
        [
            [float(row["num_linear"]), float(row["num_attention"]), float(row["num_relu"])]
            for row in rows
        ],
        dtype=np.float32,
    )
    step_features = np.asarray(
        [encode_step_features(int(row["step"]), max_step=max_step) for row in rows],
        dtype=np.float32,
    )
    step_indices = np.asarray([step_to_idx[int(row["step"])] for row in rows], dtype=np.int64)
    if target_name == "train":
        target = np.asarray([float(row["train_loss"]) for row in rows], dtype=np.float32)
    else:
        target = np.asarray([float(row["val_loss"]) for row in rows], dtype=np.float32)

    return MetaDataset(
        arch_ids=arch_ids,
        seq_tokens=seq_tokens,
        count_features=count_features,
        step_features=step_features,
        step_indices=step_indices,
        step_values=np.asarray(step_values, dtype=np.int64),
        target=target,
    )


def build_grouped_splits(architecture_ids: list[int], num_folds: int, num_repeats: int) -> list[dict[str, object]]:
    unique_ids = sorted(set(architecture_ids))
    if len(unique_ids) % num_folds != 0:
        raise SystemExit(f"number of architectures={len(unique_ids)} must be divisible by num_folds={num_folds}")
    fold_size = len(unique_ids) // num_folds
    splits = []
    for repeat in range(num_repeats):
        rng = random.Random(1000 + repeat)
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)
        folds = [shuffled[i * fold_size : (i + 1) * fold_size] for i in range(num_folds)]
        for fold_idx in range(num_folds):
            test_ids = folds[fold_idx]
            val_ids = folds[(fold_idx + 1) % num_folds]
            train_ids = [
                aid
                for idx, fold in enumerate(folds)
                if idx not in (fold_idx, (fold_idx + 1) % num_folds)
                for aid in fold
            ]
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold_idx,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                }
            )
    return splits


def build_split_masks(dataset: MetaDataset, split: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ids = set(int(x) for x in split["train_ids"])
    val_ids = set(int(x) for x in split["val_ids"])
    test_ids = set(int(x) for x in split["test_ids"])
    train_mask = np.asarray([aid in train_ids for aid in dataset.arch_ids], dtype=bool)
    val_mask = np.asarray([aid in val_ids for aid in dataset.arch_ids], dtype=bool)
    test_mask = np.asarray([aid in test_ids for aid in dataset.arch_ids], dtype=bool)
    return train_mask, val_mask, test_mask


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_step_mean_baseline_metrics(dataset: MetaDataset, split: dict[str, object]) -> dict[str, float]:
    train_mask, val_mask, test_mask = build_split_masks(dataset, split)
    baseline_all = np.zeros(len(dataset.target), dtype=np.float32)
    for step_idx in range(len(dataset.step_values)):
        mask = train_mask & (dataset.step_indices == step_idx)
        baseline_all[dataset.step_indices == step_idx] = float(dataset.target[mask].mean())

    residual_target = dataset.target - baseline_all
    target_mean = float(residual_target[train_mask].mean())
    target_std = float(residual_target[train_mask].std())
    if target_std < 1e-8:
        target_std = 1.0

    target_norm = (residual_target - target_mean) / target_std
    pred_norm = (np.zeros(len(dataset.target), dtype=np.float32) - target_mean) / target_std
    return {
        "train_mae": compute_mae(dataset.target[train_mask], baseline_all[train_mask]),
        "val_mae": compute_mae(dataset.target[val_mask], baseline_all[val_mask]),
        "test_mae": compute_mae(dataset.target[test_mask], baseline_all[test_mask]),
        "train_objective_mse": float(np.mean((pred_norm[train_mask] - target_norm[train_mask]) ** 2)),
        "val_objective_mse": float(np.mean((pred_norm[val_mask] - target_norm[val_mask]) ** 2)),
        "test_objective_mse": float(np.mean((pred_norm[test_mask] - target_norm[test_mask]) ** 2)),
    }


def train_one_split(
    dataset: MetaDataset,
    split: dict[str, object],
    *,
    device: torch.device,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    embedding_dim: int,
    step_emb_dim: int,
    torch_seed: int,
) -> tuple[list[dict[str, float]], int]:
    train_mask, val_mask, test_mask = build_split_masks(dataset, split)

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

    seed = torch_seed + 100 * int(split["repeat"]) + int(split["fold"])
    set_seed(seed)
    model = TimefitMetaMLP(
        arch_mode="counts",
        time_mode="discrete",
        num_steps=len(dataset.step_values),
        embedding_dim=embedding_dim,
        step_emb_dim=step_emb_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    curves: list[dict[str, float]] = []
    best_epoch = -1
    best_selection = float("inf")
    epochs_since_best = 0

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
            train_objective_mse = float(F.mse_loss(pred_norm[train_idx], target_tensor[train_idx]).item())
            val_objective_mse = float(F.mse_loss(pred_norm[val_idx], target_tensor[val_idx]).item())
            test_objective_mse = float(F.mse_loss(pred_norm[test_idx], target_tensor[test_idx]).item())
            pred_all = baseline_all + pred_norm.cpu().numpy() * target_std + target_mean

        curves.append(
            {
                "epoch": float(epoch),
                "train_mae": compute_mae(dataset.target[train_mask], pred_all[train_mask]),
                "val_mae": compute_mae(dataset.target[val_mask], pred_all[val_mask]),
                "test_mae": compute_mae(dataset.target[test_mask], pred_all[test_mask]),
                "train_objective_mse": train_objective_mse,
                "val_objective_mse": val_objective_mse,
                "test_objective_mse": test_objective_mse,
                "selection_loss": val_objective_mse,
            }
        )

        if val_objective_mse < best_selection - 1e-9:
            best_selection = val_objective_mse
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch + 1 >= min_epochs and epochs_since_best >= patience:
            break

    return curves, best_epoch


def summarize_metric_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_representative_split(selected_rows: list[dict[str, float]]) -> dict[str, float]:
    mean_test_mae = float(np.mean([row["test_mae"] for row in selected_rows]))
    mean_val_mae = float(np.mean([row["val_mae"] for row in selected_rows]))
    return min(
        selected_rows,
        key=lambda row: (
            abs(row["test_mae"] - mean_test_mae),
            abs(row["val_mae"] - mean_val_mae),
            int(row["repeat"]),
            int(row["fold"]),
        ),
    )


def plot_mean_curve(
    path: Path,
    *,
    curves_by_split: list[list[dict[str, float]]],
    selected_epochs: list[int],
    baseline_rows: list[dict[str, float]],
    metric_keys: list[str],
    metric_title: str,
    ylabel: str,
) -> None:
    max_len = max(len(curve) for curve in curves_by_split)
    x = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = {
        "train": "#1f77b4",
        "val": "#2a9d8f",
        "test": "#e76f51",
    }

    for metric_key in metric_keys:
        values = np.full((len(curves_by_split), max_len), np.nan, dtype=np.float64)
        for row_idx, curve in enumerate(curves_by_split):
            for point_idx, point in enumerate(curve):
                values[row_idx, point_idx] = float(point[metric_key])
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        split_name = metric_key.split("_", 1)[0]
        color = colors[split_name]
        ax.plot(x, mean, linewidth=2.0, color=color, label=f"{split_name} mean")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

        baseline_key = metric_key
        baseline_values = np.asarray([row[baseline_key] for row in baseline_rows], dtype=np.float64)
        ax.axhline(
            float(baseline_values.mean()),
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.8,
        )

    ax.axvline(float(np.mean(selected_epochs)), linestyle=":", linewidth=1.4, color="#666666", label="mean selected epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(metric_title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_vs_baseline(
    path: Path,
    *,
    baseline_rows: list[dict[str, float]],
    selected_rows: list[dict[str, float]],
) -> None:
    splits = ["train", "val", "test"]
    mae_keys = [f"{split}_mae" for split in splits]
    mse_keys = [f"{split}_objective_mse" for split in splits]
    x = np.arange(len(splits))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    baseline_means = [float(np.mean([row[key] for row in baseline_rows])) for key in mae_keys]
    baseline_stds = [float(np.std([row[key] for row in baseline_rows], ddof=0)) for key in mae_keys]
    selected_means = [float(np.mean([row[key] for row in selected_rows])) for key in mae_keys]
    selected_stds = [float(np.std([row[key] for row in selected_rows], ddof=0)) for key in mae_keys]
    ax.bar(x - width / 2, baseline_means, yerr=baseline_stds, width=width, color="#9ecae1", label="step-mean baseline")
    ax.bar(x + width / 2, selected_means, yerr=selected_stds, width=width, color="#1f77b4", label="selected checkpoint")
    ax.set_xticks(x, splits)
    ax.set_ylabel("MAE")
    ax.set_title("Original-scale MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    baseline_means = [float(np.mean([row[key] for row in baseline_rows])) for key in mse_keys]
    baseline_stds = [float(np.std([row[key] for row in baseline_rows], ddof=0)) for key in mse_keys]
    selected_means = [float(np.mean([row[key] for row in selected_rows])) for key in mse_keys]
    selected_stds = [float(np.std([row[key] for row in selected_rows], ddof=0)) for key in mse_keys]
    ax.bar(x - width / 2, baseline_means, yerr=baseline_stds, width=width, color="#f5c18a", label="step-mean baseline")
    ax.bar(x + width / 2, selected_means, yerr=selected_stds, width=width, color="#e76f51", label="selected checkpoint")
    ax.set_xticks(x, splits)
    ax.set_ylabel("objective MSE")
    ax.set_title("Standardized objective MSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Full328 heldout counts_stepmean_residual vs step-mean baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative_split(
    path: Path,
    *,
    curve: list[dict[str, float]],
    baseline: dict[str, float],
    repeat: int,
    fold: int,
    selected_epoch: int,
) -> None:
    epochs = [int(point["epoch"]) for point in curve]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    for key, color in (("train_mae", "#1f77b4"), ("val_mae", "#2a9d8f"), ("test_mae", "#e76f51")):
        ax.plot(epochs, [point[key] for point in curve], linewidth=2.0, color=color, label=key)
        ax.axhline(baseline[key], linestyle="--", linewidth=1.2, color=color, alpha=0.8)
    ax.axvline(selected_epoch, linestyle=":", linewidth=1.3, color="#666666")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE")
    ax.set_title(f"Representative split r{repeat} f{fold}: MAE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for key, color in (
        ("train_objective_mse", "#1f77b4"),
        ("val_objective_mse", "#2a9d8f"),
        ("test_objective_mse", "#e76f51"),
    ):
        ax.plot(epochs, [point[key] for point in curve], linewidth=2.0, color=color, label=key)
        ax.axhline(baseline[key], linestyle="--", linewidth=1.2, color=color, alpha=0.8)
    ax.axvline(selected_epoch, linestyle=":", linewidth=1.3, color="#666666")
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective MSE")
    ax.set_title(f"Representative split r{repeat} f{fold}: objective MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = pick_device(args.device)
    meta_csv = Path(args.meta_csv).resolve()
    dataset = read_meta_rows(meta_csv, target_name=args.target_name)
    splits = build_grouped_splits(dataset.arch_ids.tolist(), num_folds=args.num_folds, num_repeats=args.num_repeats)

    all_curve_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, float]] = []
    baseline_rows: list[dict[str, float]] = []
    curves_by_split: list[list[dict[str, float]]] = []
    selected_epochs: list[int] = []
    curve_lookup: dict[tuple[int, int], list[dict[str, float]]] = {}

    for split in splits:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        curve, best_epoch = train_one_split(
            dataset,
            split,
            device=device,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            step_emb_dim=args.step_emb_dim,
            torch_seed=args.torch_seed,
        )
        baseline_metrics = compute_step_mean_baseline_metrics(dataset, split)
        selected_metrics = dict(curve[best_epoch])
        selected_metrics["repeat"] = float(repeat)
        selected_metrics["fold"] = float(fold)
        baseline_metrics["repeat"] = float(repeat)
        baseline_metrics["fold"] = float(fold)

        selected_rows.append(selected_metrics)
        baseline_rows.append(baseline_metrics)
        curves_by_split.append(curve)
        selected_epochs.append(best_epoch)
        curve_lookup[(repeat, fold)] = curve

        for point in curve:
            all_curve_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    **point,
                }
            )

    representative = choose_representative_split(selected_rows)
    rep_repeat = int(representative["repeat"])
    rep_fold = int(representative["fold"])
    rep_curve = curve_lookup[(rep_repeat, rep_fold)]
    rep_baseline = next(
        row for row in baseline_rows if int(row["repeat"]) == rep_repeat and int(row["fold"]) == rep_fold
    )
    rep_selected_epoch = int(representative["epoch"])

    write_csv(out_dir / "full328_counts_stepmean_residual_curves.csv", all_curve_rows)
    write_csv(
        out_dir / "full328_counts_stepmean_residual_selected_vs_baseline.csv",
        [
            {"model": "selected_checkpoint", **row} for row in selected_rows
        ]
        + [
            {"model": "step_mean_baseline", **row} for row in baseline_rows
        ],
    )

    plot_mean_curve(
        out_dir / "full328_counts_stepmean_residual_mean_mae_curves.png",
        curves_by_split=curves_by_split,
        selected_epochs=selected_epochs,
        baseline_rows=baseline_rows,
        metric_keys=["train_mae", "val_mae", "test_mae"],
        metric_title="Full328 heldout counts_stepmean_residual: mean MAE curves across splits",
        ylabel="MAE",
    )
    plot_mean_curve(
        out_dir / "full328_counts_stepmean_residual_mean_objective_curves.png",
        curves_by_split=curves_by_split,
        selected_epochs=selected_epochs,
        baseline_rows=baseline_rows,
        metric_keys=["train_objective_mse", "val_objective_mse", "test_objective_mse"],
        metric_title="Full328 heldout counts_stepmean_residual: mean objective-MSE curves across splits",
        ylabel="objective MSE",
    )
    plot_selected_vs_baseline(
        out_dir / "full328_counts_stepmean_residual_selected_vs_baseline.png",
        baseline_rows=baseline_rows,
        selected_rows=selected_rows,
    )
    plot_representative_split(
        out_dir / "full328_counts_stepmean_residual_representative_split.png",
        curve=rep_curve,
        baseline=rep_baseline,
        repeat=rep_repeat,
        fold=rep_fold,
        selected_epoch=rep_selected_epoch,
    )

    metric_keys = [
        "train_mae",
        "val_mae",
        "test_mae",
        "train_objective_mse",
        "val_objective_mse",
        "test_objective_mse",
    ]
    summary = {
        "meta_csv": str(meta_csv),
        "device_resolved": str(device),
        "target_name": args.target_name,
        "num_rows": int(len(dataset.target)),
        "num_architectures": int(len(set(dataset.arch_ids.tolist()))),
        "step_values": dataset.step_values.tolist(),
        "num_folds": args.num_folds,
        "num_repeats": args.num_repeats,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "step_emb_dim": args.step_emb_dim,
        "arch_mode": "counts",
        "time_mode": "discrete",
        "target_mode": "residual",
        "input_dim": 3 + args.step_emb_dim,
        "layer_dims": [3 + args.step_emb_dim, args.hidden_dim, args.hidden_dim, 1],
        "step_embedding_shape": [len(dataset.step_values), args.step_emb_dim],
        "parameter_count": int(
            sum(
                p.numel()
                for p in TimefitMetaMLP(
                    arch_mode="counts",
                    time_mode="discrete",
                    num_steps=len(dataset.step_values),
                    embedding_dim=args.embedding_dim,
                    step_emb_dim=args.step_emb_dim,
                    hidden_dim=args.hidden_dim,
                ).parameters()
            )
        ),
        "selected_epoch_mean": float(np.mean(selected_epochs)),
        "selected_epoch_std": float(np.std(selected_epochs, ddof=0)),
        "representative_split": {
            "repeat": rep_repeat,
            "fold": rep_fold,
            "selected_epoch": rep_selected_epoch,
        },
        "selected_checkpoint_summary": summarize_metric_rows(selected_rows, metric_keys),
        "baseline_summary": summarize_metric_rows(baseline_rows, metric_keys),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[run_full328_timefit_residual] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
