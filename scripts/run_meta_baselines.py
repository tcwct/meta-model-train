from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run grouped meta-model baselines on the minimal architecture family.")
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--num_repeats", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=8)
    p.add_argument("--torch_seed", type=int, default=123)
    return p


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_meta_rows(meta_csv: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(meta_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count_fields = [field for field in (reader.fieldnames or []) if field.startswith("num_")]
        for row in reader:
            step = int(row["step"])
            if step < 0:
                continue
            item: dict[str, object] = {
                "architecture_id": int(row["architecture_id"]),
                "architecture_code": row["architecture_code"],
                "step": step,
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
            for field in count_fields:
                item[field] = float(row[field])
            rows.append(item)
    if not rows:
        raise SystemExit(f"no usable rows found in {meta_csv}")
    return rows


def build_grouped_splits(architecture_ids: list[int], num_folds: int, num_repeats: int) -> list[dict[str, object]]:
    unique_ids = sorted(set(architecture_ids))
    if num_folds < 3:
        raise SystemExit("num_folds must be >= 3 because this protocol reserves separate val and test folds")
    if len(unique_ids) < num_folds:
        raise SystemExit(f"number of architectures={len(unique_ids)} must be >= num_folds={num_folds}")
    splits: list[dict[str, object]] = []
    for repeat in range(num_repeats):
        rng = random.Random(1000 + repeat)
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)
        folds = [list(chunk) for chunk in np.array_split(np.asarray(shuffled, dtype=np.int64), num_folds)]
        for fold_idx in range(num_folds):
            test_ids = folds[fold_idx]
            val_ids = folds[(fold_idx + 1) % num_folds]
            train_ids = [aid for j, fold in enumerate(folds) if j not in (fold_idx, (fold_idx + 1) % num_folds) for aid in fold]
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


def encode_step_features(step: int, max_step: int) -> list[float]:
    if max_step <= 0:
        return [0.0, 0.0]
    step_norm = step / max_step
    log_norm = math.log1p(step) / math.log1p(max_step)
    return [step_norm, log_norm]


@dataclass
class MetaDataset:
    arch_ids: np.ndarray
    seq_tokens: np.ndarray
    count_features: np.ndarray
    count_feature_names: tuple[str, ...]
    step_features: np.ndarray
    train_target: np.ndarray
    val_target: np.ndarray
    max_step: int
    seq_len: int
    vocab_size: int


def build_meta_dataset(rows: list[dict[str, object]]) -> MetaDataset:
    token_vocab = sorted({tok for row in rows for tok in str(row["architecture_code"]).split("-")})
    token_to_id = {tok: idx for idx, tok in enumerate(token_vocab)}
    count_feature_names = tuple(sorted(key for key in rows[0].keys() if str(key).startswith("num_")))
    seq_len = len(str(rows[0]["architecture_code"]).split("-"))
    max_step = max(int(row["step"]) for row in rows)
    arch_ids = np.asarray([int(row["architecture_id"]) for row in rows], dtype=np.int64)
    seq_tokens = np.asarray(
        [[token_to_id[tok] for tok in str(row["architecture_code"]).split("-")] for row in rows],
        dtype=np.int64,
    )
    count_features = np.asarray(
        [[float(row[name]) for name in count_feature_names] for row in rows],
        dtype=np.float32,
    )
    step_features = np.asarray([encode_step_features(int(row["step"]), max_step=max_step) for row in rows], dtype=np.float32)
    train_target = np.asarray([float(row["train_loss"]) for row in rows], dtype=np.float32)[:, None]
    val_target = np.asarray([float(row["val_loss"]) for row in rows], dtype=np.float32)[:, None]
    return MetaDataset(
        arch_ids=arch_ids,
        seq_tokens=seq_tokens,
        count_features=count_features,
        count_feature_names=count_feature_names,
        step_features=step_features,
        train_target=train_target,
        val_target=val_target,
        max_step=max_step,
        seq_len=seq_len,
        vocab_size=len(token_vocab),
    )


class StepOnlyMLP(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, step_features: torch.Tensor) -> torch.Tensor:
        return self.net(step_features)


class CountsMLP(nn.Module):
    def __init__(self, count_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(count_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, count_features: torch.Tensor, step_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat((count_features, step_features), dim=-1)
        return self.net(x)


class SequenceMLP(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = seq_len * embedding_dim + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, seq_tokens: torch.Tensor, step_features: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(seq_tokens).reshape(seq_tokens.shape[0], -1)
        x = torch.cat((emb, step_features), dim=-1)
        return self.net(x)


def build_model(
    model_name: str,
    *,
    count_dim: int,
    vocab_size: int,
    seq_len: int,
    embedding_dim: int,
    hidden_dim: int,
    out_dim: int,
) -> nn.Module:
    if model_name == "step_only":
        return StepOnlyMLP(hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "counts":
        return CountsMLP(count_dim=count_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "sequence":
        return SequenceMLP(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )
    raise ValueError(f"unknown model_name={model_name}")


def model_forward(
    model: nn.Module,
    model_name: str,
    seq_tokens: torch.Tensor,
    count_features: torch.Tensor,
    step_features: torch.Tensor,
) -> torch.Tensor:
    if model_name == "step_only":
        return model(step_features)
    if model_name == "counts":
        return model(count_features, step_features)
    if model_name == "sequence":
        return model(seq_tokens, step_features)
    raise ValueError(f"unknown model_name={model_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    y_mean = float(np.mean(y_true))
    denom = float(np.sum((y_true - y_mean) ** 2))
    r2 = float(1.0 - np.sum(diff**2) / denom) if denom > 0.0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_architecture_mae(arch_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_arch = []
    for arch_id in sorted(set(int(a) for a in arch_ids.tolist())):
        mask = arch_ids == arch_id
        per_arch.append(float(np.mean(np.abs(y_pred[mask] - y_true[mask]))))
    return float(np.mean(per_arch))


def standardize_target(
    train_values: np.ndarray,
    all_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_values, axis=0, keepdims=True)
    std = np.std(train_values, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (all_values - mean) / std
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def build_split_tensors(dataset: MetaDataset, split: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    arch_ids = dataset.arch_ids
    train_ids = set(int(x) for x in split["train_ids"])
    val_ids = set(int(x) for x in split["val_ids"])
    test_ids = set(int(x) for x in split["test_ids"])

    train_mask = np.asarray([aid in train_ids for aid in arch_ids], dtype=bool)
    val_mask = np.asarray([aid in val_ids for aid in arch_ids], dtype=bool)
    test_mask = np.asarray([aid in test_ids for aid in arch_ids], dtype=bool)

    return {
        "seq_tokens": torch.from_numpy(dataset.seq_tokens).to(device=device, dtype=torch.long),
        "count_features": torch.from_numpy(dataset.count_features).to(device=device, dtype=torch.float32),
        "step_features": torch.from_numpy(dataset.step_features).to(device=device, dtype=torch.float32),
        "train_mask": torch.from_numpy(train_mask).to(device=device, dtype=torch.bool),
        "val_mask": torch.from_numpy(val_mask).to(device=device, dtype=torch.bool),
        "test_mask": torch.from_numpy(test_mask).to(device=device, dtype=torch.bool),
    }


def train_one_split(
    dataset: MetaDataset,
    split: dict[str, object],
    *,
    model_name: str,
    target_name: str,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    embedding_dim: int,
    torch_seed: int,
) -> dict[str, object]:
    set_seed(torch_seed + 97 * int(split["repeat"]) + 13 * int(split["fold"]))
    tensors = build_split_tensors(dataset, split, device)
    seq_tokens = tensors["seq_tokens"]
    count_features = tensors["count_features"]
    step_features = tensors["step_features"]
    train_mask = tensors["train_mask"]
    val_mask = tensors["val_mask"]
    test_mask = tensors["test_mask"]

    if target_name == "train":
        target_all = dataset.train_target
    elif target_name == "val":
        target_all = dataset.val_target
    elif target_name == "joint":
        target_all = np.concatenate([dataset.train_target, dataset.val_target], axis=1)
    else:
        raise ValueError(f"unknown target_name={target_name}")

    target_norm, target_mean, target_std = standardize_target(target_all[train_mask.cpu().numpy()], target_all)
    target_tensor = torch.from_numpy(target_norm).to(device=device, dtype=torch.float32)
    out_dim = int(target_tensor.shape[1])

    model = build_model(
        model_name,
        count_dim=dataset.count_features.shape[1],
        vocab_size=dataset.vocab_size,
        seq_len=dataset.seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_since_best = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model_forward(model, model_name, seq_tokens, count_features, step_features)
        train_loss = F.mse_loss(pred[train_mask], target_tensor[train_mask])
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model_forward(model, model_name, seq_tokens, count_features, step_features)
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
        pred_norm = model_forward(model, model_name, seq_tokens, count_features, step_features).cpu().numpy()
    pred_all = pred_norm * target_std + target_mean

    runtime_s = time.perf_counter() - t0
    result: dict[str, object] = {
        "repeat": split["repeat"],
        "fold": split["fold"],
        "model_name": model_name,
        "target_name": target_name,
        "best_epoch": best_epoch,
        "best_val_objective": best_val_loss,
        "runtime_s": runtime_s,
    }

    if target_name in ("train", "val"):
        true = target_all[:, 0]
        pred = pred_all[:, 0]
        for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            mask_np = mask.cpu().numpy()
            metrics = compute_metrics(true[mask_np], pred[mask_np])
            arch_mae = compute_architecture_mae(dataset.arch_ids[mask_np], true[mask_np], pred[mask_np])
            for key, value in metrics.items():
                result[f"{split_name}_{key}"] = value
            result[f"{split_name}_arch_mae"] = arch_mae
    else:
        for head_idx, head_name in enumerate(("train", "val")):
            true = target_all[:, head_idx]
            pred = pred_all[:, head_idx]
            for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
                mask_np = mask.cpu().numpy()
                metrics = compute_metrics(true[mask_np], pred[mask_np])
                arch_mae = compute_architecture_mae(dataset.arch_ids[mask_np], true[mask_np], pred[mask_np])
                for key, value in metrics.items():
                    result[f"{head_name}_{split_name}_{key}"] = value
                result[f"{head_name}_{split_name}_arch_mae"] = arch_mae

    return result


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in results:
        grouped.setdefault((str(row["model_name"]), str(row["target_name"])), []).append(row)

    summary: dict[str, object] = {}
    for key, rows in grouped.items():
        model_name, target_name = key
        metric_summary: dict[str, float] = {}
        numeric_keys = [k for k in rows[0].keys() if k not in ("model_name", "target_name")]
        for metric_key in numeric_keys:
            values = [float(row[metric_key]) for row in rows if isinstance(row[metric_key], (int, float))]
            if values:
                metric_summary[f"{metric_key}_mean"] = float(np.mean(values))
                metric_summary[f"{metric_key}_std"] = float(np.std(values))
        summary[f"{model_name}__{target_name}"] = metric_summary
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_comparison(summary: dict[str, object], output_png: Path) -> None:
    labels = []
    values = []
    errors = []
    for experiment_key in (
        "step_only__val",
        "counts__val",
        "sequence__val",
        "sequence__train",
        "sequence__joint",
    ):
        if experiment_key not in summary:
            continue
        entry = summary[experiment_key]
        if experiment_key == "sequence__joint":
            value = entry["val_test_mae_mean"]
            error = entry["val_test_mae_std"]
            label = "sequence->joint(val head)"
        else:
            value = entry["test_mae_mean"]
            error = entry["test_mae_std"]
            label = experiment_key.replace("__", " -> ")
        labels.append(label)
        values.append(value)
        errors.append(error)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("held-out architecture test MAE")
    ax.set_title("Meta-model comparison across grouped CV splits")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_joint_heads(summary: dict[str, object], output_png: Path) -> None:
    labels = ["train-only", "joint train head", "val-only", "joint val head"]
    values = [
        summary["sequence__train"]["test_mae_mean"],
        summary["sequence__joint"]["train_test_mae_mean"],
        summary["sequence__val"]["test_mae_mean"],
        summary["sequence__joint"]["val_test_mae_mean"],
    ]
    errors = [
        summary["sequence__train"]["test_mae_std"],
        summary["sequence__joint"]["train_test_mae_std"],
        summary["sequence__val"]["test_mae_std"],
        summary["sequence__joint"]["val_test_mae_std"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("held-out architecture test MAE")
    ax.set_title("Single-target vs joint-target sequence meta-models")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = pick_device(args.device)
    rows = read_meta_rows(args.meta_csv)
    dataset = build_meta_dataset(rows)
    splits = build_grouped_splits(dataset.arch_ids.tolist(), num_folds=args.num_folds, num_repeats=args.num_repeats)

    experiments = [
        ("step_only", "val"),
        ("counts", "val"),
        ("sequence", "val"),
        ("sequence", "train"),
        ("sequence", "joint"),
    ]

    manifest = {
        "meta_csv": str(Path(args.meta_csv).resolve()),
        "device_resolved": str(device),
        "num_rows": len(rows),
        "num_architectures": len(sorted(set(dataset.arch_ids.tolist()))),
        "num_folds": args.num_folds,
        "num_repeats": args.num_repeats,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "count_feature_names": list(dataset.count_feature_names),
        "token_vocab_size": dataset.vocab_size,
        "sequence_length": dataset.seq_len,
        "experiments": [{"model_name": m, "target_name": t} for m, t in experiments],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_results: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for model_name, target_name in experiments:
        print(f"[run_meta_baselines] start experiment model={model_name} target={target_name}")
        for split in splits:
            result = train_one_split(
                dataset,
                split,
                model_name=model_name,
                target_name=target_name,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                torch_seed=args.torch_seed,
            )
            all_results.append(result)
            print(
                f"[run_meta_baselines] done model={model_name} target={target_name} "
                f"repeat={result['repeat']} fold={result['fold']} best_epoch={result['best_epoch']}"
            )

    total_runtime = time.perf_counter() - t0
    write_csv(out_dir / "cv_results.csv", all_results)
    summary = summarize_results(all_results)
    summary["total_runtime_s"] = total_runtime
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(summary, out_dir / "comparison_test_mae.png")
    plot_joint_heads(summary, out_dir / "joint_vs_single_targets.png")
    print(f"[run_meta_baselines] wrote_results={out_dir}")


if __name__ == "__main__":
    main()
