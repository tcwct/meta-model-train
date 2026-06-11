from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import run_meta_baselines as baselines


MAIN_METRIC = "test_log_mae_step_ge_100"
SECONDARY_METRICS = (
    "test_log_mae",
    "test_log_rmse",
    "test_spearman_step_300",
    "test_spearman_step_999",
    "test_mae",
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep residual log-loss meta-model capacity on the formal_v2 dataset.")
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--num_folds", type=int, default=3)
    p.add_argument("--num_repeats", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--confirm_max_epochs", type=int, default=1500)
    p.add_argument("--confirm_patience", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--torch_seed", type=int, default=123)
    p.add_argument("--counts_hidden_dims", type=str, default="2,4,8,12,16")
    p.add_argument("--sequence_configs", type=str, default="2x2,4x2,8x2,4x3,8x3,4x4")
    p.add_argument("--gru_embedding_dims", type=str, default="2,4")
    p.add_argument("--gru_hidden_dims", type=str, default="4,8,16")
    p.add_argument("--sequence_gru_head_dim", type=int, default=8)
    p.add_argument("--direct_counts_hidden_dims", type=str, default="2,4,6,8")
    p.add_argument("--direct_sequence_configs", type=str, default="2x2,3x2,4x2,6x2,4x3")
    p.add_argument("--topk_per_family", type=int, default=2)
    return p


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise SystemExit("expected at least one integer value")
    return values


def parse_pair_list(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" not in chunk:
            raise SystemExit(f"expected pairs like 4x2, got {chunk!r}")
        hidden_str, embedding_str = chunk.split("x", 1)
        pairs.append((int(hidden_str), int(embedding_str)))
    if not pairs:
        raise SystemExit("expected at least one hidden/embedding pair")
    return pairs


def count_model_parameters(
    dataset: baselines.MetaDataset,
    *,
    model_name: str,
    hidden_dim: int,
    embedding_dim: int,
    sequence_gru_head_dim: int,
) -> int:
    model = baselines.build_model(
        model_name,
        count_dim=dataset.count_features.shape[1],
        vocab_size=dataset.vocab_size,
        seq_len=dataset.seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        out_dim=1,
        sequence_gru_head_dim=sequence_gru_head_dim,
    )
    return int(sum(param.numel() for param in model.parameters()))


def build_configs(dataset: baselines.MetaDataset, args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = [
        {
            "config_id": "step_mean_log",
            "family": "baseline_step_mean",
            "model_name": "step_mean",
            "hidden_dim": None,
            "embedding_dim": None,
            "param_count": 0,
            "target_mode": "direct",
            "target_transform": "log",
            "sequence_gru_head_dim": args.sequence_gru_head_dim,
            "label": "step_mean (log prior only)",
        }
    ]

    for hidden_dim in parse_int_list(args.counts_hidden_dims):
        params = count_model_parameters(
            dataset,
            model_name="counts",
            hidden_dim=hidden_dim,
            embedding_dim=1,
            sequence_gru_head_dim=args.sequence_gru_head_dim,
        )
        configs.append(
            {
                "config_id": f"residual_counts_h{hidden_dim}",
                "family": "residual_counts",
                "model_name": "counts",
                "hidden_dim": hidden_dim,
                "embedding_dim": None,
                "param_count": params,
                "target_mode": "residual_over_step_mean",
                "target_transform": "log",
                "sequence_gru_head_dim": args.sequence_gru_head_dim,
                "label": f"residual counts h={hidden_dim} ({params}p)",
            }
        )

    for hidden_dim, embedding_dim in parse_pair_list(args.sequence_configs):
        params = count_model_parameters(
            dataset,
            model_name="sequence",
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            sequence_gru_head_dim=args.sequence_gru_head_dim,
        )
        configs.append(
            {
                "config_id": f"residual_sequence_h{hidden_dim}_e{embedding_dim}",
                "family": "residual_sequence_mlp",
                "model_name": "sequence",
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "param_count": params,
                "target_mode": "residual_over_step_mean",
                "target_transform": "log",
                "sequence_gru_head_dim": args.sequence_gru_head_dim,
                "label": f"residual seq-mlp h={hidden_dim} e={embedding_dim} ({params}p)",
            }
        )

    for embedding_dim in parse_int_list(args.gru_embedding_dims):
        for hidden_dim in parse_int_list(args.gru_hidden_dims):
            params = count_model_parameters(
                dataset,
                model_name="sequence_gru",
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                sequence_gru_head_dim=args.sequence_gru_head_dim,
            )
            configs.append(
                {
                    "config_id": f"residual_gru_h{hidden_dim}_e{embedding_dim}",
                    "family": "residual_sequence_gru",
                    "model_name": "sequence_gru",
                    "hidden_dim": hidden_dim,
                    "embedding_dim": embedding_dim,
                    "param_count": params,
                    "target_mode": "residual_over_step_mean",
                    "target_transform": "log",
                    "sequence_gru_head_dim": args.sequence_gru_head_dim,
                    "label": f"residual seq-gru h={hidden_dim} e={embedding_dim} ({params}p)",
                }
            )

    for hidden_dim in parse_int_list(args.direct_counts_hidden_dims):
        params = count_model_parameters(
            dataset,
            model_name="counts",
            hidden_dim=hidden_dim,
            embedding_dim=1,
            sequence_gru_head_dim=args.sequence_gru_head_dim,
        )
        configs.append(
            {
                "config_id": f"direct_counts_h{hidden_dim}",
                "family": "direct_counts_aux",
                "model_name": "counts",
                "hidden_dim": hidden_dim,
                "embedding_dim": None,
                "param_count": params,
                "target_mode": "direct",
                "target_transform": "log",
                "sequence_gru_head_dim": args.sequence_gru_head_dim,
                "label": f"direct counts h={hidden_dim} ({params}p)",
            }
        )

    for hidden_dim, embedding_dim in parse_pair_list(args.direct_sequence_configs):
        params = count_model_parameters(
            dataset,
            model_name="sequence",
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            sequence_gru_head_dim=args.sequence_gru_head_dim,
        )
        configs.append(
            {
                "config_id": f"direct_sequence_h{hidden_dim}_e{embedding_dim}",
                "family": "direct_sequence_aux",
                "model_name": "sequence",
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "param_count": params,
                "target_mode": "direct",
                "target_transform": "log",
                "sequence_gru_head_dim": args.sequence_gru_head_dim,
                "label": f"direct seq-mlp h={hidden_dim} e={embedding_dim} ({params}p)",
            }
        )
    return configs


def summarize_config_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    ignore = {
        "config_id",
        "family",
        "model_name",
        "label",
        "repeat",
        "fold",
        "target_mode",
        "target_transform",
    }
    metric_keys = [key for key in rows[0].keys() if key not in ignore]
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def choose_representative_row(rows: list[dict[str, object]], metric_key: str = MAIN_METRIC) -> dict[str, object]:
    target = float(np.mean([float(row[metric_key]) for row in rows]))
    return min(rows, key=lambda row: abs(float(row[metric_key]) - target))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def fit_with_history(
    dataset: baselines.MetaDataset,
    split: dict[str, object],
    *,
    model_name: str,
    hidden_dim: int,
    embedding_dim: int,
    target_mode: str,
    target_transform: str,
    sequence_gru_head_dim: int,
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

    target_all_raw = baselines.select_target_array(dataset, "val")
    target_all_model = baselines.transform_target_array(target_all_raw, target_transform)
    train_mask_np = train_mask.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
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
        out_dim=1,
        sequence_gru_head_dim=sequence_gru_head_dim,
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

    pred_all_model = pred_norm * target_std + target_mean + baseline_all
    pred_all_raw = baselines.inverse_target_array(pred_all_model, target_transform)
    true = target_all_raw[:, 0]
    pred = pred_all_raw[:, 0]
    metrics: dict[str, float] = {}
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        mask_np = mask.cpu().numpy()
        split_metrics = baselines.compute_extended_metrics(
            arch_ids=dataset.arch_ids,
            steps=dataset.steps,
            y_true_raw=true,
            y_pred_raw=pred,
            mask_np=mask_np,
        )
        for key, value in split_metrics.items():
            metrics[f"{split_name}_{key}"] = value
        metrics[f"{split_name}_objective_mse"] = float(np.mean((pred_norm[mask_np] - target_norm[mask_np]) ** 2))

    return {
        "best_epoch": best_epoch,
        "best_val_objective": best_val_objective,
        "history": history,
        "metrics": metrics,
    }


def plot_family_scatter(summary_rows: list[dict[str, object]], output_png: Path, baseline_value: float) -> None:
    families = [
        ("residual_counts", "Residual Counts"),
        ("residual_sequence_mlp", "Residual Sequence MLP"),
        ("residual_sequence_gru", "Residual Sequence GRU"),
        ("direct_counts_aux", "Direct Counts (Aux)"),
        ("direct_sequence_aux", "Direct Sequence MLP (Aux)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes_list = list(axes.flatten())
    cmap = plt.get_cmap("tab10")
    for ax, (family, title) in zip(axes_list, families):
        subset = [row for row in summary_rows if row["family"] == family]
        if not subset:
            ax.set_visible(False)
            continue
        for idx, row in enumerate(sorted(subset, key=lambda item: float(item["param_count_mean"]))):
            color = cmap(idx % 10)
            ax.errorbar(
                float(row["param_count_mean"]),
                float(row[f"{MAIN_METRIC}_mean"]),
                yerr=float(row[f"{MAIN_METRIC}_std"]),
                fmt="o",
                color=color,
            )
            ax.annotate(str(row["short_label"]), (float(row["param_count_mean"]), float(row[f"{MAIN_METRIC}_mean"])), fontsize=8)
        ax.axhline(baseline_value, color="black", linestyle="--", linewidth=1.2, label="step_mean")
        ax.set_title(title)
        ax.set_xlabel("parameter count")
        ax.grid(alpha=0.25)
    axes_list[0].set_ylabel("test MAE(log loss), step>=100")
    axes_list[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_bar(summary_rows: list[dict[str, object]], output_png: Path) -> None:
    selected = [row for row in summary_rows if row["family"] in {"baseline_step_mean", "residual_counts", "residual_sequence_mlp", "residual_sequence_gru", "direct_counts_aux", "direct_sequence_aux"}]
    best_rows: list[dict[str, object]] = []
    for family in ("baseline_step_mean", "residual_counts", "residual_sequence_mlp", "residual_sequence_gru", "direct_counts_aux", "direct_sequence_aux"):
        family_rows = [row for row in selected if row["family"] == family]
        if not family_rows:
            continue
        best_rows.append(min(family_rows, key=lambda row: float(row[f"{MAIN_METRIC}_mean"])))
    labels = [str(row["label"]) for row in best_rows]
    values = [float(row[f"{MAIN_METRIC}_mean"]) for row in best_rows]
    errors = [float(row[f"{MAIN_METRIC}_std"]) for row in best_rows]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("test MAE(log loss), step>=100")
    ax.set_title("Best config per family")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative_histories(representatives: list[dict[str, object]], output_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    cmap = plt.get_cmap("Dark2")
    for idx, rep in enumerate(representatives):
        color = cmap(idx % 8)
        epochs = [row["epoch"] for row in rep["history"]]
        train_curve = [row["train_objective"] for row in rep["history"]]
        val_curve = [row["val_objective"] for row in rep["history"]]
        label = str(rep["short_label"])
        ax.plot(epochs, train_curve, color=color, linewidth=1.8, label=f"{label} train")
        ax.plot(epochs, val_curve, color=color, linewidth=1.8, linestyle="--", label=f"{label} val")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective MSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_short_label(row: dict[str, object]) -> str:
    family = str(row["family"])
    if family == "baseline_step_mean":
        return "step_mean"
    if family.endswith("counts") or family == "direct_counts_aux":
        return f"h={row['hidden_dim']}"
    return f"h={row['hidden_dim']},e={row['embedding_dim']}"


def run_phase(
    *,
    dataset: baselines.MetaDataset,
    splits: list[dict[str, object]],
    configs: list[dict[str, object]],
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    torch_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for config in configs:
        print(f"[sweep_residual_log_capacity] start {config['config_id']} label={config['label']}")
        config_rows: list[dict[str, object]] = []
        for split in splits:
            row = baselines.train_one_split(
                dataset,
                split,
                model_name=str(config["model_name"]),
                target_name="val",
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                hidden_dim=int(config["hidden_dim"] or 1),
                embedding_dim=int(config["embedding_dim"] or 1),
                torch_seed=torch_seed,
                target_mode=str(config["target_mode"]),
                target_transform=str(config["target_transform"]),
                sequence_gru_head_dim=int(config["sequence_gru_head_dim"]),
            )
            merged = dict(config)
            merged.update(row)
            merged["short_label"] = add_short_label(merged)
            results.append(merged)
            config_rows.append(merged)
        summary = summarize_config_rows(config_rows)
        summary_row = dict(config)
        summary_row.update(summary)
        summary_row["short_label"] = add_short_label(summary_row)
        summary_rows.append(summary_row)
    return results, summary_rows


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = baselines.ensure_dir(args.output_dir)
    device = baselines.pick_device(args.device)
    rows = baselines.read_meta_rows(args.meta_csv)
    dataset = baselines.build_meta_dataset(rows)
    splits = baselines.build_grouped_splits(dataset.arch_ids.tolist(), num_folds=args.num_folds, num_repeats=args.num_repeats)
    configs = build_configs(dataset, args)
    config_lookup = {str(config["config_id"]): config for config in configs}

    manifest = {
        "meta_csv": str(Path(args.meta_csv).resolve()),
        "device_resolved": str(device),
        "num_rows": len(rows),
        "num_architectures": len(sorted(set(dataset.arch_ids.tolist()))),
        "num_folds": args.num_folds,
        "num_repeats": args.num_repeats,
        "phase1": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
        },
        "phase2_confirm": {
            "max_epochs": args.confirm_max_epochs,
            "patience": args.confirm_patience,
            "topk_per_family": args.topk_per_family,
        },
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sequence_gru_head_dim": args.sequence_gru_head_dim,
        "configs": configs,
        "main_metric": MAIN_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    phase1_results, phase1_summary = run_phase(
        dataset=dataset,
        splits=splits,
        configs=configs,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        torch_seed=args.torch_seed,
    )
    baselines.write_csv(out_dir / "phase1_cv_results.csv", phase1_results)
    write_summary_csv(out_dir / "phase1_summary.csv", phase1_summary)

    baseline_row = next(row for row in phase1_summary if row["family"] == "baseline_step_mean")
    plot_family_scatter(phase1_summary, out_dir / "phase1_capacity_scatter.png", float(baseline_row[f"{MAIN_METRIC}_mean"]))
    plot_best_bar(phase1_summary, out_dir / "phase1_best_family_bar.png")

    confirm_candidates: list[dict[str, object]] = []
    for family in ("residual_counts", "residual_sequence_mlp", "residual_sequence_gru"):
        subset = [row for row in phase1_summary if row["family"] == family]
        subset = sorted(subset, key=lambda row: float(row[f"{MAIN_METRIC}_mean"]))
        confirm_candidates.extend(config_lookup[str(row["config_id"])] for row in subset[: args.topk_per_family])

    confirm_results, confirm_summary = run_phase(
        dataset=dataset,
        splits=splits,
        configs=confirm_candidates,
        device=device,
        max_epochs=args.confirm_max_epochs,
        patience=args.confirm_patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        torch_seed=args.torch_seed,
    )
    baselines.write_csv(out_dir / "phase2_confirm_cv_results.csv", confirm_results)
    write_summary_csv(out_dir / "phase2_confirm_summary.csv", confirm_summary)

    representatives: list[dict[str, object]] = []
    for family in ("residual_counts", "residual_sequence_mlp", "residual_sequence_gru", "direct_counts_aux", "direct_sequence_aux"):
        family_rows = [row for row in phase1_summary if row["family"] == family]
        if not family_rows:
            continue
        best_summary = min(family_rows, key=lambda row: float(row[f"{MAIN_METRIC}_mean"]))
        source_rows = [row for row in phase1_results if row["config_id"] == best_summary["config_id"]]
        representative_row = choose_representative_row(source_rows)
        history_fit = fit_with_history(
            dataset,
            {
                "repeat": representative_row["repeat"],
                "fold": representative_row["fold"],
                "train_ids": next(split["train_ids"] for split in splits if split["repeat"] == representative_row["repeat"] and split["fold"] == representative_row["fold"]),
                "val_ids": next(split["val_ids"] for split in splits if split["repeat"] == representative_row["repeat"] and split["fold"] == representative_row["fold"]),
                "test_ids": next(split["test_ids"] for split in splits if split["repeat"] == representative_row["repeat"] and split["fold"] == representative_row["fold"]),
            },
            model_name=str(best_summary["model_name"]),
            hidden_dim=int(best_summary["hidden_dim"] or 1),
            embedding_dim=int(best_summary["embedding_dim"] or 1),
            target_mode=str(best_summary["target_mode"]),
            target_transform=str(best_summary["target_transform"]),
            sequence_gru_head_dim=int(best_summary["sequence_gru_head_dim"]),
            device=device,
            max_epochs=args.confirm_max_epochs if str(best_summary["family"]).startswith("residual_") else args.max_epochs,
            patience=args.confirm_patience if str(best_summary["family"]).startswith("residual_") else args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            torch_seed=args.torch_seed,
        )
        rep = dict(best_summary)
        rep.update(
            {
                "history": history_fit["history"],
                "representative_repeat": representative_row["repeat"],
                "representative_fold": representative_row["fold"],
            }
        )
        rep.update(history_fit["metrics"])
        representatives.append(rep)

    (out_dir / "representative_histories.json").write_text(json.dumps(representatives, indent=2), encoding="utf-8")
    plot_representative_histories(
        [row for row in representatives if str(row["family"]).startswith("residual_")],
        out_dir / "residual_representative_histories.png",
        title="Residual Log-Loss: Representative Train/Val Curves",
    )
    plot_representative_histories(
        [row for row in representatives if str(row["family"]).startswith("direct_")],
        out_dir / "direct_representative_histories.png",
        title="Direct Log-Loss: Representative Train/Val Curves",
    )

    phase1_best = {
        family: min(
            [row for row in phase1_summary if row["family"] == family],
            key=lambda row: float(row[f"{MAIN_METRIC}_mean"]),
        )
        for family in ("baseline_step_mean", "residual_counts", "residual_sequence_mlp", "residual_sequence_gru", "direct_counts_aux", "direct_sequence_aux")
    }
    confirm_best = {
        family: min(
            [row for row in confirm_summary if row["family"] == family],
            key=lambda row: float(row[f"{MAIN_METRIC}_mean"]),
        )
        for family in ("residual_counts", "residual_sequence_mlp", "residual_sequence_gru")
    }

    report = {
        "main_metric": MAIN_METRIC,
        "phase1_best": phase1_best,
        "phase2_confirm_best": confirm_best,
        "baseline_main_metric": phase1_best["baseline_step_mean"][f"{MAIN_METRIC}_mean"],
        "total_runtime_s": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sweep_residual_log_capacity] wrote_results={out_dir}")
    print(f"[sweep_residual_log_capacity] total_runtime_s={report['total_runtime_s']:.3f}")


if __name__ == "__main__":
    main()
