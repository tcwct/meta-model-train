from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore and compare full328 v1/v3 meta datasets and experiment outputs.")
    parser.add_argument(
        "--family_root",
        type=str,
        default="outputs/toy_diffusion/meta_model_family",
    )
    parser.add_argument(
        "--experiments_root",
        type=str,
        default="outputs/toy_diffusion/meta_model_experiments",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/toy_diffusion/meta_model_experiments/full328_v1_v3_exploration",
    )
    return parser


def ensure_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(meta_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with meta_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
                    "num_linear": int(row["num_linear"]),
                    "num_attention": int(row["num_attention"]),
                    "num_relu": int(row["num_relu"]),
                }
            )
    if not rows:
        raise ValueError(f"no usable rows found in {meta_csv}")
    return rows


def build_curve_matrix(rows: list[dict[str, object]], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    curves_by_arch: dict[int, dict[int, float]] = defaultdict(dict)
    steps = sorted({int(row["step"]) for row in rows})
    step_to_col = {step: idx for idx, step in enumerate(steps)}
    for row in rows:
        curves_by_arch[int(row["architecture_id"])][int(row["step"])] = float(row[metric_key])

    matrix = np.full((len(curves_by_arch), len(steps)), np.nan, dtype=np.float64)
    for row_idx, arch_id in enumerate(sorted(curves_by_arch)):
        for step, value in curves_by_arch[arch_id].items():
            matrix[row_idx, step_to_col[step]] = value
    return np.asarray(steps, dtype=np.int64), matrix


def summarize_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(matrix, axis=0),
        "q10": np.nanpercentile(matrix, 10, axis=0),
        "q25": np.nanpercentile(matrix, 25, axis=0),
        "q75": np.nanpercentile(matrix, 75, axis=0),
        "q90": np.nanpercentile(matrix, 90, axis=0),
    }


def grouped_splits(architecture_ids: list[int], num_folds: int = 4, num_repeats: int = 3) -> list[dict[str, object]]:
    unique_ids = sorted(set(architecture_ids))
    if len(unique_ids) % num_folds != 0:
        raise ValueError(f"number of architectures={len(unique_ids)} must be divisible by num_folds={num_folds}")
    fold_size = len(unique_ids) // num_folds
    splits: list[dict[str, object]] = []
    for repeat in range(num_repeats):
        rng = random.Random(1000 + repeat)
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)
        folds = [shuffled[i * fold_size : (i + 1) * fold_size] for i in range(num_folds)]
        for fold_idx in range(num_folds):
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold_idx,
                    "train_ids": [
                        arch_id
                        for j, fold in enumerate(folds)
                        if j not in (fold_idx, (fold_idx + 1) % num_folds)
                        for arch_id in fold
                    ],
                    "val_ids": folds[(fold_idx + 1) % num_folds],
                    "test_ids": folds[fold_idx],
                }
            )
    return splits


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_empirical_baselines(rows: list[dict[str, object]]) -> dict[str, object]:
    splits = grouped_splits([int(row["architecture_id"]) for row in rows], num_folds=4, num_repeats=3)
    split_rows: list[dict[str, float]] = []
    for split in splits:
        train_ids = set(int(x) for x in split["train_ids"])
        test_ids = set(int(x) for x in split["test_ids"])
        train_rows = [row for row in rows if int(row["architecture_id"]) in train_ids]
        test_rows = [row for row in rows if int(row["architecture_id"]) in test_ids]

        step_means: dict[int, float] = {}
        step_groups: dict[int, list[float]] = defaultdict(list)
        for row in train_rows:
            step_groups[int(row["step"])].append(float(row["val_loss"]))
        for step, values in step_groups.items():
            step_means[step] = float(np.mean(values))

        step_count_means: dict[tuple[int, tuple[int, int, int]], float] = {}
        step_count_groups: dict[tuple[int, tuple[int, int, int]], list[float]] = defaultdict(list)
        for row in train_rows:
            key = (
                int(row["step"]),
                (int(row["num_linear"]), int(row["num_attention"]), int(row["num_relu"])),
            )
            step_count_groups[key].append(float(row["val_loss"]))
        for key, values in step_count_groups.items():
            step_count_means[key] = float(np.mean(values))

        y_true = np.asarray([float(row["val_loss"]) for row in test_rows], dtype=np.float64)
        y_step = np.asarray([step_means[int(row["step"])] for row in test_rows], dtype=np.float64)
        y_count = np.asarray(
            [
                step_count_means.get(
                    (
                        int(row["step"]),
                        (int(row["num_linear"]), int(row["num_attention"]), int(row["num_relu"])),
                    ),
                    step_means[int(row["step"])],
                )
                for row in test_rows
            ],
            dtype=np.float64,
        )
        split_rows.append(
            {
                "repeat": float(split["repeat"]),
                "fold": float(split["fold"]),
                "step_only_test_mae": mae(y_true, y_step),
                "step_count_group_test_mae": mae(y_true, y_count),
            }
        )

    step_values = np.asarray([row["step_only_test_mae"] for row in split_rows], dtype=np.float64)
    count_values = np.asarray([row["step_count_group_test_mae"] for row in split_rows], dtype=np.float64)
    return {
        "split_rows": split_rows,
        "step_only_test_mae_mean": float(np.mean(step_values)),
        "step_only_test_mae_std": float(np.std(step_values)),
        "step_count_group_test_mae_mean": float(np.mean(count_values)),
        "step_count_group_test_mae_std": float(np.std(count_values)),
        "count_better_splits": int(np.sum(count_values < step_values)),
        "num_splits": int(len(split_rows)),
    }


def compute_per_step_empirical_baselines(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    splits = grouped_splits([int(row["architecture_id"]) for row in rows], num_folds=4, num_repeats=3)
    steps = sorted({int(row["step"]) for row in rows})
    out: list[dict[str, float]] = []
    for step in steps:
        step_only_values: list[float] = []
        count_values: list[float] = []
        for split in splits:
            train_ids = set(int(x) for x in split["train_ids"])
            test_ids = set(int(x) for x in split["test_ids"])
            train_rows = [
                row for row in rows if int(row["architecture_id"]) in train_ids and int(row["step"]) == step
            ]
            test_rows = [
                row for row in rows if int(row["architecture_id"]) in test_ids and int(row["step"]) == step
            ]
            step_mean = float(np.mean([float(row["val_loss"]) for row in train_rows]))
            step_count_groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
            for row in train_rows:
                key = (
                    int(row["num_linear"]),
                    int(row["num_attention"]),
                    int(row["num_relu"]),
                )
                step_count_groups[key].append(float(row["val_loss"]))
            step_count_means = {key: float(np.mean(values)) for key, values in step_count_groups.items()}

            y_true = np.asarray([float(row["val_loss"]) for row in test_rows], dtype=np.float64)
            y_step = np.asarray([step_mean for _ in test_rows], dtype=np.float64)
            y_count = np.asarray(
                [
                    step_count_means.get(
                        (
                            int(row["num_linear"]),
                            int(row["num_attention"]),
                            int(row["num_relu"]),
                        ),
                        step_mean,
                    )
                    for row in test_rows
                ],
                dtype=np.float64,
            )
            step_only_values.append(mae(y_true, y_step))
            count_values.append(mae(y_true, y_count))

        out.append(
            {
                "step": float(step),
                "progress": float(step / steps[-1]),
                "step_only_test_mae_mean": float(np.mean(step_only_values)),
                "step_only_test_mae_std": float(np.std(step_only_values)),
                "step_count_group_test_mae_mean": float(np.mean(count_values)),
                "step_count_group_test_mae_std": float(np.std(count_values)),
                "absolute_gain_mean": float(np.mean(step_only_values) - np.mean(count_values)),
            }
        )
    total_step_mae = float(sum(row["step_only_test_mae_mean"] for row in out))
    for row in out:
        row["step_only_mae_share"] = float(row["step_only_test_mae_mean"] / total_step_mae) if total_step_mae > 0 else 0.0
    return out


def architecture_signature(row: dict[str, object], mode: str) -> tuple[object, ...]:
    code = str(row["architecture_code"]).split("-")
    if mode == "counts":
        return (
            int(row["num_linear"]),
            int(row["num_attention"]),
            int(row["num_relu"]),
        )
    if mode == "first_last":
        return (code[0], code[-1])
    if mode == "half_counts":
        left = code[:3]
        right = code[3:]
        return (
            left.count("L"),
            left.count("A"),
            left.count("R"),
            right.count("L"),
            right.count("A"),
            right.count("R"),
        )
    if mode == "transition_counts":
        transitions = tuple(a + b for a, b in zip(code[:-1], code[1:]))
        types = [a + b for a in ("L", "A", "R") for b in ("L", "A", "R")]
        return tuple(transitions.count(t) for t in types)
    if mode == "counts_plus_transitions":
        return architecture_signature(row, "counts") + architecture_signature(row, "transition_counts")
    raise ValueError(f"unknown signature mode={mode}")


def compute_signature_baseline_sweep(rows: list[dict[str, object]]) -> list[dict[str, float | str]]:
    modes = ["counts", "first_last", "half_counts", "transition_counts", "counts_plus_transitions"]
    splits = grouped_splits([int(row["architecture_id"]) for row in rows], num_folds=4, num_repeats=3)
    out: list[dict[str, float | str]] = []
    for mode in modes:
        split_maes: list[float] = []
        for split in splits:
            train_ids = set(int(x) for x in split["train_ids"])
            test_ids = set(int(x) for x in split["test_ids"])
            train_rows = [row for row in rows if int(row["architecture_id"]) in train_ids]
            test_rows = [row for row in rows if int(row["architecture_id"]) in test_ids]

            step_means: dict[int, float] = {}
            step_groups: dict[int, list[float]] = defaultdict(list)
            for row in train_rows:
                step_groups[int(row["step"])].append(float(row["val_loss"]))
            for step, values in step_groups.items():
                step_means[step] = float(np.mean(values))

            grouped_means: dict[tuple[int, tuple[object, ...]], float] = {}
            grouped_lists: dict[tuple[int, tuple[object, ...]], list[float]] = defaultdict(list)
            for row in train_rows:
                key = (int(row["step"]), architecture_signature(row, mode))
                grouped_lists[key].append(float(row["val_loss"]))
            for key, values in grouped_lists.items():
                grouped_means[key] = float(np.mean(values))

            y_true = np.asarray([float(row["val_loss"]) for row in test_rows], dtype=np.float64)
            y_pred = np.asarray(
                [
                    grouped_means.get((int(row["step"]), architecture_signature(row, mode)), step_means[int(row["step"])])
                    for row in test_rows
                ],
                dtype=np.float64,
            )
            split_maes.append(mae(y_true, y_pred))

        out.append(
            {
                "mode": mode,
                "test_mae_mean": float(np.mean(split_maes)),
                "test_mae_std": float(np.std(split_maes)),
            }
        )
    return out


def compute_within_step_count_r2(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    steps = sorted({int(row["step"]) for row in rows})
    out: list[dict[str, float]] = []
    for step in steps:
        subset = [row for row in rows if int(row["step"]) == step]
        y = np.asarray([float(row["val_loss"]) for row in subset], dtype=np.float64)
        y_mean = float(np.mean(y))
        sst = float(np.sum((y - y_mean) ** 2))

        groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        for row in subset:
            groups[
                (int(row["num_linear"]), int(row["num_attention"]), int(row["num_relu"]))
            ].append(float(row["val_loss"]))
        group_means = {key: float(np.mean(values)) for key, values in groups.items()}
        y_pred = np.asarray(
            [
                group_means[(int(row["num_linear"]), int(row["num_attention"]), int(row["num_relu"]))]
                for row in subset
            ],
            dtype=np.float64,
        )
        sse = float(np.sum((y - y_pred) ** 2))
        out.append(
            {
                "step": float(step),
                "progress": float(step / steps[-1]),
                "count_group_r2": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
            }
        )
    return out


def compute_step_to_final_correlation(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    by_arch: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_arch[int(row["architecture_id"])][int(row["step"])] = float(row["val_loss"])
    steps = sorted(next(iter(by_arch.values())).keys())
    final_step = steps[-1]
    final_values = np.asarray([by_arch[arch_id][final_step] for arch_id in sorted(by_arch)], dtype=np.float64)

    out: list[dict[str, float]] = []
    for step in steps:
        x = np.asarray([by_arch[arch_id][step] for arch_id in sorted(by_arch)], dtype=np.float64)
        corr = float(np.corrcoef(x, final_values)[0, 1])
        out.append(
            {
                "step": float(step),
                "progress": float(step / final_step),
                "corr_with_final": corr,
            }
        )
    return out


def compute_dataset_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    steps = sorted({int(row["step"]) for row in rows})
    final_step = steps[-1]
    final_rows = [row for row in rows if int(row["step"]) == final_step]
    final_val = np.asarray([float(row["val_loss"]) for row in final_rows], dtype=np.float64)
    final_train = np.asarray([float(row["train_loss"]) for row in final_rows], dtype=np.float64)

    count_groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in final_rows:
        key = (int(row["num_linear"]), int(row["num_attention"]), int(row["num_relu"]))
        count_groups[key].append(float(row["val_loss"]))

    final_group_rows = []
    for key, values in count_groups.items():
        val_arr = np.asarray(values, dtype=np.float64)
        final_group_rows.append(
            {
                "count_signature": f"L{key[0]}-A{key[1]}-R{key[2]}",
                "num_linear": key[0],
                "num_attention": key[1],
                "num_relu": key[2],
                "n_architectures": int(len(values)),
                "mean_final_val": float(np.mean(val_arr)),
                "std_final_val": float(np.std(val_arr)),
                "cv_final_val": float(np.std(val_arr) / np.mean(val_arr)) if float(np.mean(val_arr)) > 0 else float("nan"),
            }
        )
    final_group_rows.sort(key=lambda row: float(row["mean_final_val"]))

    return {
        "num_rows": int(len(rows)),
        "num_architectures": int(len({int(row["architecture_id"]) for row in rows})),
        "num_steps": int(len(steps)),
        "step_values": steps,
        "final_step": int(final_step),
        "final_val_mean": float(np.mean(final_val)),
        "final_val_median": float(np.median(final_val)),
        "final_val_min": float(np.min(final_val)),
        "final_val_max": float(np.max(final_val)),
        "final_train_mean": float(np.mean(final_train)),
        "num_count_signatures": int(len(count_groups)),
        "best_count_groups": final_group_rows[:5],
        "worst_count_groups": final_group_rows[-5:],
    }


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


def plot_val_quantiles(path: Path, curve_stats: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = {"v1": "#1f77b4", "v3": "#d55e00"}
    for name, stats in curve_stats.items():
        color = palette[name]
        ax.fill_between(stats["progress"], stats["q10"], stats["q90"], color=color, alpha=0.12)
        ax.fill_between(stats["progress"], stats["q25"], stats["q75"], color=color, alpha=0.22)
        ax.plot(stats["progress"], stats["median"], color=color, linewidth=2.5, label=name)
    ax.set_xlabel("training progress (step / max_step)")
    ax.set_ylabel("validation loss")
    ax.set_yscale("log")
    ax.set_title("Validation-loss quantiles across architectures")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_count_r2(path: Path, stats_by_name: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = {"v1": "#1f77b4", "v3": "#d55e00"}
    for name, rows in stats_by_name.items():
        ax.plot(
            [row["progress"] for row in rows],
            [row["count_group_r2"] for row in rows],
            marker="o",
            linewidth=2.0,
            color=palette[name],
            label=name,
        )
    ax.set_xlabel("training progress (step / max_step)")
    ax.set_ylabel("within-step R^2 of count-group means")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("How much do count signatures explain within-step variation?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_corr_with_final(path: Path, stats_by_name: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = {"v1": "#1f77b4", "v3": "#d55e00"}
    for name, rows in stats_by_name.items():
        ax.plot(
            [row["progress"] for row in rows],
            [row["corr_with_final"] for row in rows],
            marker="o",
            linewidth=2.0,
            color=palette[name],
            label=name,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#666666")
    ax.set_xlabel("training progress (step / max_step)")
    ax.set_ylabel("Pearson correlation with final-step val loss")
    ax.set_ylim(-0.15, 1.02)
    ax.set_title("When does the final architecture ranking become visible?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_bars(
    path: Path,
    grouped_cv_summaries: dict[str, dict[str, object]],
    empirical_baselines: dict[str, dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=False)
    bar_names = [
        ("learned step_only", "step_only__val", "#4c78a8"),
        ("learned counts", "counts__val", "#72b7b2"),
        ("learned sequence", "sequence__val", "#e45756"),
        ("empirical step", None, "#9d9d9d"),
        ("empirical step+count", None, "#54a24b"),
    ]
    for ax, dataset_name in zip(axes, ("v1", "v3")):
        summary = grouped_cv_summaries[dataset_name]
        empirical = empirical_baselines[dataset_name]
        means = [
            float(summary["step_only__val"]["test_mae_mean"]),
            float(summary["counts__val"]["test_mae_mean"]),
            float(summary["sequence__val"]["test_mae_mean"]),
            float(empirical["step_only_test_mae_mean"]),
            float(empirical["step_count_group_test_mae_mean"]),
        ]
        stds = [
            float(summary["step_only__val"]["test_mae_std"]),
            float(summary["counts__val"]["test_mae_std"]),
            float(summary["sequence__val"]["test_mae_std"]),
            float(empirical["step_only_test_mae_std"]),
            float(empirical["step_count_group_test_mae_std"]),
        ]
        x = np.arange(len(bar_names))
        ax.bar(x, means, yerr=stds, capsize=3, color=[item[2] for item in bar_names])
        ax.set_xticks(x, [item[0] for item in bar_names], rotation=20, ha="right")
        ax.set_ylabel("held-out architecture test MAE")
        ax.set_title(dataset_name)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out architecture generalization: learned models vs empirical baselines")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_step_mae_shares(path: Path, stats_by_name: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=False)
    palette = {"v1": "#1f77b4", "v3": "#d55e00"}
    for ax, dataset_name in zip(axes, ("v1", "v3")):
        rows = stats_by_name[dataset_name]
        ax.bar(
            [row["step"] for row in rows],
            [row["step_only_mae_share"] for row in rows],
            width=35,
            color=palette[dataset_name],
            alpha=0.85,
        )
        ax.set_xlabel("training step")
        ax.set_ylabel("share of total raw step-only MAE")
        ax.set_title(dataset_name)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Which training steps dominate raw MAE?")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_step_count_gain(path: Path, stats_by_name: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    palette = {"v1": "#1f77b4", "v3": "#d55e00"}
    for dataset_name, rows in stats_by_name.items():
        ax.plot(
            [row["progress"] for row in rows],
            [row["absolute_gain_mean"] for row in rows],
            marker="o",
            linewidth=2.0,
            color=palette[dataset_name],
            label=dataset_name,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#666666")
    ax.set_xlabel("training progress (step / max_step)")
    ax.set_ylabel("step baseline MAE minus step+count MAE")
    ax.set_title("Where does count information help on the raw loss scale?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(
    output_path: Path,
    dataset_summaries: dict[str, dict[str, object]],
    grouped_cv_summaries: dict[str, dict[str, object]],
    residual_summaries: dict[str, dict[str, object]],
    empirical_baselines: dict[str, dict[str, object]],
) -> None:
    v1 = dataset_summaries["v1"]
    v3 = dataset_summaries["v3"]
    grouped_v1 = grouped_cv_summaries["v1"]
    grouped_v3 = grouped_cv_summaries["v3"]
    residual_v1 = residual_summaries["v1"]
    residual_v3 = residual_summaries["v3"]
    empirical_v1 = empirical_baselines["v1"]
    empirical_v3 = empirical_baselines["v3"]

    lines = [
        "# Full328 v1/v3 exploration",
        "",
        "## Scope",
        "",
        "- Compare the old `server_v1_full328` dataset against the new `server_v3_full328_residual1000` dataset.",
        "- Revisit held-out architecture generalization with both the trained baselines and cheap empirical baselines.",
        "- Look for structure in the data itself before jumping to bigger models.",
        "",
        "## High-level takeaways",
        "",
        f"- The first-stage conclusion still holds: on held-out architectures, learned `step_only` remains better than learned `counts`, and learned `sequence` is still worst on both v1 and v3.",
        f"- The residual time-fit result also still holds: the selected checkpoint improves objective-space MSE but does not reliably beat the step-mean baseline on original-scale test MAE.",
        f"- New and important nuance: `counts` information is not useless. A simple train-split lookup baseline using `step + count signature` beats the pure step baseline on {empirical_v1['count_better_splits']}/{empirical_v1['num_splits']} v1 splits and {empirical_v3['count_better_splits']}/{empirical_v3['num_splits']} v3 splits.",
        "- That suggests the current learned `counts` MLP is under-using available count information, rather than proving that count information itself has no value.",
        "- Another new pattern is that v3 keeps much more within-count-group diversity at late training steps. In v1, many count groups nearly collapse to the same final loss; in v3, sequence/order effects inside the same count signature remain much larger.",
        "- Visual inspection of a representative v3 residual split also suggests a time-modeling problem: predicted curves are much more jagged and less monotone than the true curves.",
        "- In v3, early-step validation loss already correlates strongly with final-step ranking, unlike v1. So architecture signal is present very early, even if the current held-out predictors do not exploit it well.",
        "- A very important evaluation caveat emerged: raw MAE is dominated by the earliest training steps. For the empirical step baseline, step 0 alone contributes about 86% of total v3 MAE, and the first three steps already dominate v1 as well.",
        "- A final negative result is also useful: naive order-aware hard signatures such as `first+last token`, `half-counts`, or `transition-counts` do not beat plain count signatures as grouped-mean baselines. They seem to become too sparse too quickly.",
        "",
        "## Dataset summary",
        "",
        f"- v1: {v1['num_architectures']} architectures, {v1['num_steps']} usable time points, final step {v1['final_step']}, final val mean {v1['final_val_mean']:.6f}, median {v1['final_val_median']:.6f}.",
        f"- v3: {v3['num_architectures']} architectures, {v3['num_steps']} usable time points, final step {v3['final_step']}, final val mean {v3['final_val_mean']:.6f}, median {v3['final_val_median']:.6f}.",
        "",
        "## Held-out generalization numbers",
        "",
        "| Dataset | learned step_only | learned counts | learned sequence | empirical step | empirical step+count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| v1 | {grouped_v1['step_only__val']['test_mae_mean']:.6f} | {grouped_v1['counts__val']['test_mae_mean']:.6f} | {grouped_v1['sequence__val']['test_mae_mean']:.6f} | {empirical_v1['step_only_test_mae_mean']:.6f} | {empirical_v1['step_count_group_test_mae_mean']:.6f} |",
        f"| v3 | {grouped_v3['step_only__val']['test_mae_mean']:.6f} | {grouped_v3['counts__val']['test_mae_mean']:.6f} | {grouped_v3['sequence__val']['test_mae_mean']:.6f} | {empirical_v3['step_only_test_mae_mean']:.6f} | {empirical_v3['step_count_group_test_mae_mean']:.6f} |",
        "",
        "Interpretation:",
        "",
        "- The learning problem on v3 is harder in absolute MAE terms than on v1.",
        "- But raw MAE is not an equal-step metric. On v3 especially, it mostly measures whether a model gets the very beginning of training right.",
        "- But the empirical `step+count` baseline becoming stronger on v3 means the harder problem is not simply a lack of architecture signal.",
        "- Instead, it looks more like a representation/generalization problem: useful structure exists, but the current learned models are not extracting it robustly under grouped CV.",
        "",
        "## Residual time-fit line",
        "",
        f"- v1 baseline vs selected checkpoint test MAE: {residual_v1['baseline_summary']['test_mae_mean']:.6f} vs {residual_v1['selected_checkpoint_summary']['test_mae_mean']:.6f}.",
        f"- v1 baseline vs selected checkpoint test objective MSE: {residual_v1['baseline_summary']['test_objective_mse_mean']:.6f} vs {residual_v1['selected_checkpoint_summary']['test_objective_mse_mean']:.6f}.",
        f"- v3 baseline vs selected checkpoint test MAE: {residual_v3['baseline_summary']['test_mae_mean']:.6f} vs {residual_v3['selected_checkpoint_summary']['test_mae_mean']:.6f}.",
        f"- v3 baseline vs selected checkpoint test objective MSE: {residual_v3['baseline_summary']['test_objective_mse_mean']:.6f} vs {residual_v3['selected_checkpoint_summary']['test_objective_mse_mean']:.6f}.",
        "",
        "Interpretation:",
        "",
        "- The residual model is learning something real in normalized residual space.",
        "- But that improvement does not transfer cleanly to the original loss scale, especially on v3.",
        "- This strengthens the earlier suspicion that matching residual objective values is easier than improving the metric we actually care about on unseen architectures.",
        "- The representative prediction curves are also visibly rougher than the real curves, which is consistent with the current discrete-time model having weak smoothness bias.",
        "- Because raw MAE is heavily front-loaded, a model can improve the part of the curve we may care about scientifically and still fail to win the headline MAE metric.",
        "",
        "## Count-group observations",
        "",
        "- Best v3 count groups are still linear-heavy; worst v3 count groups are attention-heavy or mixed attention/relu groups.",
        "- Compared with v1, late-stage v3 losses are much smaller overall, but within the same count signature they are more spread out relative to the mean.",
        "- So at 1000 steps, count totals alone do not determine the eventual curve nearly as tightly as they seemed to at 500 steps.",
        "- The `v1_v3_corr_with_final.png` plot adds another angle: v3 architecture rankings are already partially visible near the beginning of training, while v1 rankings emerge much later.",
        "- The per-step baseline decomposition shows that count information helps most on the raw loss scale very early in v3, especially at step 0. Late-step gains exist but are numerically tiny because the late losses themselves are tiny.",
        "",
        "## Caution",
        "",
        "- The empirical `step+count` baseline is a lookup-style grouped average, not a learned parametric model. It proves that the information exists, not that a more flexible neural model will automatically use it.",
        "- The failure of harder grouped signatures does not mean order is irrelevant. It may simply mean that these signatures partition the space too finely and need shared parameters rather than exact matching.",
        "- Correlation and variance-explained plots are descriptive diagnostics, not causal evidence about why training behaves this way.",
        "- The jagged-curve observation comes from one representative split, so it should be treated as a qualitative clue rather than a stand-alone conclusion.",
        "- The MAE-dominance analysis is exact for the empirical baselines we computed here, but extending the same weighting conclusion to every learned model should still be stated carefully unless their per-step errors are inspected directly.",
        "",
        "## Suggested next moves",
        "",
        "1. Add `empirical step+count-group` as an official baseline in the repo. Right now it is informative enough that future learned models should beat it, not just the pure step baseline.",
        "2. Attack time representation separately from architecture representation. The new v3 data makes it less plausible that the earlier result was only caused by sparse time sampling.",
        "3. Report at least one step-balanced metric alongside raw MAE. Otherwise v3 evaluation is overwhelmingly controlled by the first one or two time points.",
        "4. Try structured architecture features before larger sequence models: position-aware counts, transition counts, front-half/back-half counts, or per-position token indicators.",
        "5. Keep conclusions conservative: the current evidence says the problem remains hard, not that architecture order is irrelevant.",
        "",
        "## Outputs",
        "",
        "- `v1_v3_val_quantiles.png`",
        "- `v1_v3_count_group_r2.png`",
        "- `v1_v3_corr_with_final.png`",
        "- `v1_v3_generalization_bars.png`",
        "- `v1_v3_step_mae_shares.png`",
        "- `v1_v3_per_step_count_gain.png`",
        "- `dataset_summaries.json`",
        "- `empirical_grouped_baselines.json`",
        "- `signature_baseline_sweep.json`",
        "- `../v3_pred_curves/r0f0_pred_vs_actual.png`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    family_root = Path(args.family_root).resolve()
    experiments_root = Path(args.experiments_root).resolve()
    out_dir = ensure_dir(args.output_dir)

    datasets = {
        "v1": load_rows(family_root / "server_v1_full328_meta_dataset.csv"),
        "v3": load_rows(family_root / "server_v3_full328_residual1000_meta_dataset.csv"),
    }

    grouped_cv_summaries = {
        "v1": read_json(experiments_root / "server_v1_full328_grouped_cv" / "summary.json"),
        "v3": read_json(experiments_root / "server_v3_full328_residual1000_grouped_cv" / "summary.json"),
    }
    residual_summaries = {
        "v1": read_json(experiments_root / "server_v1_full328_timefit_counts_stepmean_residual_2500" / "summary.json"),
        "v3": read_json(experiments_root / "server_v3_full328_residual1000_timefit_counts_stepmean_residual_2500" / "summary.json"),
    }

    dataset_summaries = {name: compute_dataset_summary(rows) for name, rows in datasets.items()}
    empirical_baselines = {name: compute_empirical_baselines(rows) for name, rows in datasets.items()}
    per_step_empirical = {name: compute_per_step_empirical_baselines(rows) for name, rows in datasets.items()}
    signature_sweep = {name: compute_signature_baseline_sweep(rows) for name, rows in datasets.items()}
    within_step_count_r2 = {name: compute_within_step_count_r2(rows) for name, rows in datasets.items()}
    step_to_final_corr = {name: compute_step_to_final_correlation(rows) for name, rows in datasets.items()}

    curve_stats: dict[str, dict[str, np.ndarray]] = {}
    for name, rows in datasets.items():
        steps, val_matrix = build_curve_matrix(rows, metric_key="val_loss")
        stats = summarize_matrix(val_matrix)
        stats["steps"] = steps.astype(np.float64)
        stats["progress"] = steps.astype(np.float64) / float(steps[-1])
        curve_stats[name] = stats

    plot_val_quantiles(out_dir / "v1_v3_val_quantiles.png", curve_stats)
    plot_count_r2(out_dir / "v1_v3_count_group_r2.png", within_step_count_r2)
    plot_corr_with_final(out_dir / "v1_v3_corr_with_final.png", step_to_final_corr)
    plot_generalization_bars(out_dir / "v1_v3_generalization_bars.png", grouped_cv_summaries, empirical_baselines)
    plot_step_mae_shares(out_dir / "v1_v3_step_mae_shares.png", per_step_empirical)
    plot_per_step_count_gain(out_dir / "v1_v3_per_step_count_gain.png", per_step_empirical)

    (out_dir / "dataset_summaries.json").write_text(
        json.dumps(dataset_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "empirical_grouped_baselines.json").write_text(
        json.dumps(empirical_baselines, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "per_step_empirical_grouped_baselines.json").write_text(
        json.dumps(per_step_empirical, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "signature_baseline_sweep.json").write_text(
        json.dumps(signature_sweep, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "within_step_count_r2.json").write_text(
        json.dumps(within_step_count_r2, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "step_to_final_corr.json").write_text(
        json.dumps(step_to_final_corr, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, summary in dataset_summaries.items():
        write_csv(out_dir / f"{name}_best_count_groups.csv", list(summary["best_count_groups"]))
        write_csv(out_dir / f"{name}_worst_count_groups.csv", list(summary["worst_count_groups"]))

    build_report(
        out_dir / "report.md",
        dataset_summaries=dataset_summaries,
        grouped_cv_summaries=grouped_cv_summaries,
        residual_summaries=residual_summaries,
        empirical_baselines=empirical_baselines,
    )

    print(f"[explore_full328_v1_v3] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
