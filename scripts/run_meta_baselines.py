from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


ARCH_TOKEN_TO_ID = {"L": 0, "A": 1, "R": 2}


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


def parse_architecture_code(code: str) -> list[int]:
    return [ARCH_TOKEN_TO_ID[tok] for tok in code.split("-")]


def read_meta_rows(meta_csv: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(meta_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if step < 0:
                continue
            rows.append(
                {
                    "architecture_id": int(row["architecture_id"]),
                    "architecture_code": row["architecture_code"],
                    "step": step,
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                    "num_linear": float(row["num_linear"]),
                    "num_attention": float(row["num_attention"]),
                    "num_relu": float(row["num_relu"]),
                }
            )
    if not rows:
        raise SystemExit(f"no usable rows found in {meta_csv}")
    return rows


def build_grouped_splits(architecture_ids: list[int], num_folds: int, num_repeats: int) -> list[dict[str, object]]:
    unique_ids = sorted(set(architecture_ids))
    if len(unique_ids) % num_folds != 0:
        raise SystemExit(f"number of architectures={len(unique_ids)} must be divisible by num_folds={num_folds}")
    fold_size = len(unique_ids) // num_folds
    splits: list[dict[str, object]] = []
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
                for j, fold in enumerate(folds)
                if j not in (fold_idx, (fold_idx + 1) % num_folds)
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


def encode_step_features(step: int, max_step: int) -> list[float]:
    step_norm = step / max_step
    log_norm = math.log1p(step) / math.log1p(max_step)
    return [step_norm, log_norm]


@dataclass
class MetaDataset:
    arch_ids: np.ndarray
    seq_tokens: np.ndarray
    count_features: np.ndarray
    step_features: np.ndarray
    train_target: np.ndarray
    val_target: np.ndarray
    max_step: int


def build_meta_dataset(rows: list[dict[str, object]]) -> MetaDataset:
    max_step = max(int(row["step"]) for row in rows)
    arch_ids = np.asarray([int(row["architecture_id"]) for row in rows], dtype=np.int64)
    seq_tokens = np.asarray([parse_architecture_code(str(row["architecture_code"])) for row in rows], dtype=np.int64)
    count_features = np.asarray(
        [[float(row["num_linear"]), float(row["num_attention"]), float(row["num_relu"])] for row in rows],
        dtype=np.float32,
    )
    step_features = np.asarray([encode_step_features(int(row["step"]), max_step=max_step) for row in rows], dtype=np.float32)
    train_target = np.asarray([float(row["train_loss"]) for row in rows], dtype=np.float32)[:, None]
    val_target = np.asarray([float(row["val_loss"]) for row in rows], dtype=np.float32)[:, None]
    return MetaDataset(
        arch_ids=arch_ids,
        seq_tokens=seq_tokens,
        count_features=count_features,
        step_features=step_features,
        train_target=train_target,
        val_target=val_target,
        max_step=max_step,
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
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, count_features: torch.Tensor, step_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat((count_features, step_features), dim=-1)
        return self.net(x)


class SequenceMLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
        input_dim = 6 * embedding_dim + 2
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


def build_model(model_name: str, embedding_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    if model_name == "step_only":
        return StepOnlyMLP(hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "counts":
        return CountsMLP(hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "sequence":
        return SequenceMLP(embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim)
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
    repeat = int(split["repeat"])
    fold = int(split["fold"])
    train_ids = set(int(x) for x in split["train_ids"])
    val_ids = set(int(x) for x in split["val_ids"])
    test_ids = set(int(x) for x in split["test_ids"])

    train_mask = np.asarray([aid in train_ids for aid in dataset.arch_ids], dtype=bool)
    val_mask = np.asarray([aid in val_ids for aid in dataset.arch_ids], dtype=bool)
    test_mask = np.asarray([aid in test_ids for aid in dataset.arch_ids], dtype=bool)

    if target_name == "train":
        target_all = dataset.train_target
        out_dim = 1
    elif target_name == "val":
        target_all = dataset.val_target
        out_dim = 1
    elif target_name == "joint":
        target_all = np.concatenate((dataset.train_target, dataset.val_target), axis=1)
        out_dim = 2
    else:
        raise ValueError(f"unknown target_name={target_name}")

    train_target = target_all[train_mask]
    target_mean = train_target.mean(axis=0, keepdims=True)
    target_std = train_target.std(axis=0, keepdims=True)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)

    seq_tokens = torch.from_numpy(dataset.seq_tokens).to(device=device, dtype=torch.long)
    count_features = torch.from_numpy(dataset.count_features).to(device=device, dtype=torch.float32)
    step_features = torch.from_numpy(dataset.step_features).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy((target_all - target_mean) / target_std).to(device=device, dtype=torch.float32)

    train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
    val_idx = torch.from_numpy(np.where(val_mask)[0]).to(device)
    test_idx = torch.from_numpy(np.where(test_mask)[0]).to(device)

    set_seed(torch_seed + 100 * repeat + fold)
    model = build_model(model_name=model_name, embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_best = 0

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model_forward(model, model_name, seq_tokens, count_features, step_features)
        loss = F.mse_loss(pred[train_idx], target_tensor[train_idx])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model_forward(model, model_name, seq_tokens, count_features, step_features)
            val_loss = float(F.mse_loss(pred[val_idx], target_tensor[val_idx]).item())
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        "repeat": repeat,
        "fold": fold,
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
            metrics = compute_metrics(true[mask], pred[mask])
            arch_mae = compute_architecture_mae(dataset.arch_ids[mask], true[mask], pred[mask])
            for key, value in metrics.items():
                result[f"{split_name}_{key}"] = value
            result[f"{split_name}_arch_mae"] = arch_mae
    else:
        for head_idx, head_name in enumerate(("train", "val")):
            true = target_all[:, head_idx]
            pred = pred_all[:, head_idx]
            for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
                metrics = compute_metrics(true[mask], pred[mask])
                arch_mae = compute_architecture_mae(dataset.arch_ids[mask], true[mask], pred[mask])
                for key, value in metrics.items():
                    result[f"{head_name}_{split_name}_{key}"] = value
                result[f"{head_name}_{split_name}_arch_mae"] = arch_mae

    return result


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in results:
        grouped.setdefault((str(row["model_name"]), str(row["target_name"])), []).append(row)

    summary: dict[str, object] = {}
    for model_name, target_name in grouped:
        rows = grouped[(model_name, target_name)]
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

