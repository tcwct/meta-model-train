from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import run_meta_baselines as baselines


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep direct-mode meta-model capacity on the formal_v2 dataset.")
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--num_folds", type=int, default=3)
    p.add_argument("--num_repeats", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--torch_seed", type=int, default=123)
    p.add_argument("--hidden_dims", type=str, default="4,8,16,32,64")
    p.add_argument("--sequence_embedding_dims", type=str, default="2,4,8")
    return p


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise SystemExit("expected at least one integer value")
    return values


def count_model_parameters(dataset: baselines.MetaDataset, *, model_name: str, hidden_dim: int, embedding_dim: int) -> int:
    model = baselines.build_model(
        model_name,
        count_dim=dataset.count_features.shape[1],
        vocab_size=dataset.vocab_size,
        seq_len=dataset.seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        out_dim=1,
    )
    return int(sum(param.numel() for param in model.parameters()))


def build_configs(dataset: baselines.MetaDataset, hidden_dims: list[int], sequence_embedding_dims: list[int]) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for hidden_dim in hidden_dims:
        params = count_model_parameters(dataset, model_name="counts", hidden_dim=hidden_dim, embedding_dim=1)
        configs.append(
            {
                "config_id": f"counts_h{hidden_dim}",
                "model_name": "counts",
                "hidden_dim": hidden_dim,
                "embedding_dim": None,
                "param_count": params,
                "label": f"counts h={hidden_dim} ({params}p)",
            }
        )
    for embedding_dim in sequence_embedding_dims:
        for hidden_dim in hidden_dims:
            params = count_model_parameters(
                dataset,
                model_name="sequence",
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
            )
            configs.append(
                {
                    "config_id": f"sequence_h{hidden_dim}_e{embedding_dim}",
                    "model_name": "sequence",
                    "hidden_dim": hidden_dim,
                    "embedding_dim": embedding_dim,
                    "param_count": params,
                    "label": f"sequence h={hidden_dim} e={embedding_dim} ({params}p)",
                }
            )
    return configs


def summarize_config_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    metric_keys = [key for key in rows[0].keys() if key not in {"config_id", "model_name", "label", "repeat", "fold", "embedding_dim"}]
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if isinstance(row[key], (int, float))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def choose_representative_row(rows: list[dict[str, object]], metric_key: str = "test_mae") -> dict[str, object]:
    target = float(np.mean([float(row[metric_key]) for row in rows]))
    return min(rows, key=lambda row: abs(float(row[metric_key]) - target))


def fit_with_history(
    dataset: baselines.MetaDataset,
    split: dict[str, object],
    *,
    model_name: str,
    hidden_dim: int,
    embedding_dim: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    torch_seed: int,
) -> dict[str, object]:
    baselines.set_seed(torch_seed + 97 * int(split["repeat"]) + 13 * int(split["fold"]))
    tensors = baselines.build_split_tensors(dataset, split, device)
    seq_tokens = tensors["seq_tokens"]
    count_features = tensors["count_features"]
    step_features = tensors["step_features"]
    train_mask = tensors["train_mask"]
    val_mask = tensors["val_mask"]
    test_mask = tensors["test_mask"]

    target_all = baselines.select_target_array(dataset, "val")
    model_target_all = target_all
    train_mask_np = train_mask.cpu().numpy()
    target_norm, target_mean, target_std = baselines.standardize_target(model_target_all[train_mask_np], model_target_all)
    target_tensor = torch.from_numpy(target_norm).to(device=device, dtype=torch.float32)

    model = baselines.build_model(
        model_name,
        count_dim=dataset.count_features.shape[1],
        vocab_size=dataset.vocab_size,
        seq_len=dataset.seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        out_dim=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = -1
    best_val_objective = float("inf")
    epochs_since_best = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = baselines.model_forward(model, model_name, seq_tokens, count_features, step_features)
        train_loss = F.mse_loss(pred[train_mask], target_tensor[train_mask])
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = baselines.model_forward(model, model_name, seq_tokens, count_features, step_features)
            val_loss = float(F.mse_loss(pred[val_mask], target_tensor[val_mask]).item())
            test_loss = float(F.mse_loss(pred[test_mask], target_tensor[test_mask]).item())
            train_loss_eval = float(F.mse_loss(pred[train_mask], target_tensor[train_mask]).item())
        history.append(
            {
                "epoch": float(epoch),
                "train_objective": train_loss_eval,
                "val_objective": val_loss,
                "test_objective": test_loss,
            }
        )

        if val_loss < best_val_objective:
            best_val_objective = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
    pred_all = pred_norm * target_std + target_mean

    true = target_all[:, 0]
    pred = pred_all[:, 0]
    metrics: dict[str, float] = {}
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        mask_np = mask.cpu().numpy()
        split_metrics = baselines.compute_metrics(true[mask_np], pred[mask_np])
        for key, value in split_metrics.items():
            metrics[f"{split_name}_{key}"] = value
        metrics[f"{split_name}_arch_mae"] = baselines.compute_architecture_mae(dataset.arch_ids[mask_np], true[mask_np], pred[mask_np])
        metrics[f"{split_name}_objective_mse"] = float(np.mean((pred_norm[mask_np] - target_norm[mask_np]) ** 2))

    return {
        "best_epoch": best_epoch,
        "best_val_objective": best_val_objective,
        "history": history,
        "metrics": metrics,
    }


def plot_counts_curves(representatives: list[dict[str, object]], output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("viridis")
    ordered = sorted(representatives, key=lambda item: int(item["hidden_dim"]))
    for idx, rep in enumerate(ordered):
        color = cmap(idx / max(1, len(ordered) - 1))
        epochs = [row["epoch"] for row in rep["history"]]
        train_curve = [row["train_objective"] for row in rep["history"]]
        val_curve = [row["val_objective"] for row in rep["history"]]
        label = f"h={rep['hidden_dim']} ({rep['param_count']}p)"
        ax.plot(epochs, train_curve, color=color, linewidth=1.8, label=f"{label} train")
        ax.plot(epochs, val_curve, color=color, linewidth=1.8, linestyle="--", label=f"{label} val")
    ax.set_title("Counts Direct: Representative Train/Val Curves")
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective MSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_curves(representatives: list[dict[str, object]], output_png: Path) -> None:
    embedding_dims = sorted(set(int(rep["embedding_dim"]) for rep in representatives))
    fig, axes = plt.subplots(1, len(embedding_dims), figsize=(5.4 * len(embedding_dims), 5.0), sharey=True)
    if len(embedding_dims) == 1:
        axes = [axes]
    cmap = plt.get_cmap("plasma")
    for ax, embedding_dim in zip(axes, embedding_dims):
        subset = sorted(
            [rep for rep in representatives if int(rep["embedding_dim"]) == embedding_dim],
            key=lambda item: int(item["hidden_dim"]),
        )
        for idx, rep in enumerate(subset):
            color = cmap(idx / max(1, len(subset) - 1))
            epochs = [row["epoch"] for row in rep["history"]]
            train_curve = [row["train_objective"] for row in rep["history"]]
            val_curve = [row["val_objective"] for row in rep["history"]]
            label = f"h={rep['hidden_dim']} ({rep['param_count']}p)"
            ax.plot(epochs, train_curve, color=color, linewidth=1.7, label=f"{label} train")
            ax.plot(epochs, val_curve, color=color, linewidth=1.7, linestyle="--", label=f"{label} val")
        ax.set_title(f"Sequence Direct: e={embedding_dim}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=1)
    axes[0].set_ylabel("objective MSE")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotate_points(ax: plt.Axes, xs: list[float], ys: list[float], labels: list[str]) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)


def plot_summary_panels(summary_rows: list[dict[str, object]], output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    style = {
        "counts": ("#1f77b4", "o"),
        "sequence": ("#d62728", "s"),
    }

    for model_name in ("counts", "sequence"):
        subset = [row for row in summary_rows if row["model_name"] == model_name]
        xs = [float(row["param_count"]) for row in subset]
        ys_mae = [float(row["test_mae_mean"]) for row in subset]
        ys_gap = [float(row["test_mae_mean"]) - float(row["train_mae_mean"]) for row in subset]
        labels = [str(row["short_label"]) for row in subset]
        color, marker = style[model_name]
        axes[0].scatter(xs, ys_mae, color=color, marker=marker, s=48, label=model_name)
        axes[1].scatter(xs, ys_gap, color=color, marker=marker, s=48, label=model_name)
        annotate_points(axes[0], xs, ys_mae, labels)
        annotate_points(axes[1], xs, ys_gap, labels)

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_title("Test MAE vs Parameter Count")
    axes[1].set_title("Overfitting Gap vs Parameter Count")
    axes[0].set_xlabel("parameter count")
    axes[1].set_xlabel("parameter count")
    axes[0].set_ylabel("held-out test MAE")
    axes[1].set_ylabel("test MAE - train MAE")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_effect(summary_rows: list[dict[str, object]], output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    counts_rows = sorted([row for row in summary_rows if row["model_name"] == "counts"], key=lambda row: int(row["hidden_dim"]))
    axes[0].plot(
        [int(row["hidden_dim"]) for row in counts_rows],
        [float(row["test_mae_mean"]) for row in counts_rows],
        marker="o",
        color="#1f77b4",
    )
    for row in counts_rows:
        axes[0].annotate(f"{int(row['param_count'])}p", (int(row["hidden_dim"]), float(row["test_mae_mean"])), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_title("Counts: Hidden Dim Sweep")
    axes[0].set_xlabel("hidden_dim")
    axes[0].set_ylabel("held-out test MAE")
    axes[0].grid(alpha=0.25)

    sequence_rows = sorted(
        [row for row in summary_rows if row["model_name"] == "sequence"],
        key=lambda row: (int(row["embedding_dim"]), int(row["hidden_dim"])),
    )
    for embedding_dim in sorted(set(int(row["embedding_dim"]) for row in sequence_rows)):
        subset = [row for row in sequence_rows if int(row["embedding_dim"]) == embedding_dim]
        axes[1].plot(
            [int(row["hidden_dim"]) for row in subset],
            [float(row["test_mae_mean"]) for row in subset],
            marker="o",
            label=f"e={embedding_dim}",
        )
        for row in subset:
            axes[1].annotate(f"{int(row['param_count'])}p", (int(row["hidden_dim"]), float(row["test_mae_mean"])), xytext=(4, 4), textcoords="offset points", fontsize=6)
    axes[1].set_title("Sequence: Hidden Dim Sweep")
    axes[1].set_xlabel("hidden_dim")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = baselines.ensure_dir(args.output_dir)
    device = baselines.pick_device(args.device)
    hidden_dims = parse_int_list(args.hidden_dims)
    sequence_embedding_dims = parse_int_list(args.sequence_embedding_dims)

    rows = baselines.read_meta_rows(args.meta_csv)
    dataset = baselines.build_meta_dataset(rows)
    splits = baselines.build_grouped_splits(dataset.arch_ids.tolist(), num_folds=args.num_folds, num_repeats=args.num_repeats)
    configs = build_configs(dataset, hidden_dims, sequence_embedding_dims)

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
        "torch_seed": args.torch_seed,
        "hidden_dims": hidden_dims,
        "sequence_embedding_dims": sequence_embedding_dims,
        "target_mode": "direct",
        "target_name": "val",
        "configs": configs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    split_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    t0 = time.perf_counter()
    for config in configs:
        print(f"[sweep_direct_capacity] start {config['config_id']} label={config['label']}")
        config_results: list[dict[str, object]] = []
        for split in splits:
            result = baselines.train_one_split(
                dataset,
                split,
                model_name=str(config["model_name"]),
                target_name="val",
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden_dim=int(config["hidden_dim"]),
                embedding_dim=int(config["embedding_dim"] or 1),
                torch_seed=args.torch_seed,
                target_mode="direct",
            )
            row = {
                "config_id": str(config["config_id"]),
                "model_name": str(config["model_name"]),
                "label": str(config["label"]),
                "hidden_dim": int(config["hidden_dim"]),
                "embedding_dim": "" if config["embedding_dim"] is None else int(config["embedding_dim"]),
                "param_count": int(config["param_count"]),
                **result,
            }
            config_results.append(row)
            split_rows.append(row)

        rep = choose_representative_row(config_results, metric_key="test_mae")
        rep_split = {
            "repeat": int(rep["repeat"]),
            "fold": int(rep["fold"]),
            "train_ids": splits[0]["train_ids"],
            "val_ids": splits[0]["val_ids"],
            "test_ids": splits[0]["test_ids"],
        }
        for split in splits:
            if int(split["repeat"]) == int(rep["repeat"]) and int(split["fold"]) == int(rep["fold"]):
                rep_split = split
                break
        rep_fit = fit_with_history(
            dataset,
            rep_split,
            model_name=str(config["model_name"]),
            hidden_dim=int(config["hidden_dim"]),
            embedding_dim=int(config["embedding_dim"] or 1),
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            torch_seed=args.torch_seed,
        )
        representative_rows.append(
            {
                "config_id": str(config["config_id"]),
                "model_name": str(config["model_name"]),
                "label": str(config["label"]),
                "short_label": f"h={int(config['hidden_dim'])}" if config["model_name"] == "counts" else f"h={int(config['hidden_dim'])},e={int(config['embedding_dim'])}",
                "hidden_dim": int(config["hidden_dim"]),
                "embedding_dim": None if config["embedding_dim"] is None else int(config["embedding_dim"]),
                "param_count": int(config["param_count"]),
                "repeat": int(rep["repeat"]),
                "fold": int(rep["fold"]),
                "history": rep_fit["history"],
                "best_epoch": int(rep_fit["best_epoch"]),
                "best_val_objective": float(rep_fit["best_val_objective"]),
                **rep_fit["metrics"],
            }
        )

        summary = summarize_config_rows(config_results)
        summary_row = {
            "config_id": str(config["config_id"]),
            "model_name": str(config["model_name"]),
            "label": str(config["label"]),
            "short_label": f"h={int(config['hidden_dim'])}" if config["model_name"] == "counts" else f"h={int(config['hidden_dim'])},e={int(config['embedding_dim'])}",
            "hidden_dim": int(config["hidden_dim"]),
            "embedding_dim": "" if config["embedding_dim"] is None else int(config["embedding_dim"]),
            "param_count": int(config["param_count"]),
            **summary,
        }
        summary_row["test_minus_train_mae_mean"] = float(summary_row["test_mae_mean"] - summary_row["train_mae_mean"])
        summary_row["val_minus_train_mae_mean"] = float(summary_row["val_mae_mean"] - summary_row["train_mae_mean"])
        summary_rows.append(summary_row)

    total_runtime = time.perf_counter() - t0

    write_csv(out_dir / "split_results.csv", split_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps({"total_runtime_s": total_runtime, "rows": summary_rows}, indent=2), encoding="utf-8")
    (out_dir / "representative_histories.json").write_text(json.dumps(representative_rows, indent=2), encoding="utf-8")

    plot_counts_curves([row for row in representative_rows if row["model_name"] == "counts"], out_dir / "counts_representative_curves.png")
    plot_sequence_curves([row for row in representative_rows if row["model_name"] == "sequence"], out_dir / "sequence_representative_curves.png")
    plot_summary_panels(summary_rows, out_dir / "capacity_summary_panels.png")
    plot_hidden_effect(summary_rows, out_dir / "hidden_dim_effect.png")

    print(f"[sweep_direct_capacity] wrote_results={out_dir}")
    print(f"[sweep_direct_capacity] total_runtime_s={total_runtime:.3f}")


if __name__ == "__main__":
    main()
