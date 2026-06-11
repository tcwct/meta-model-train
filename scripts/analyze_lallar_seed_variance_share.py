from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_lallar_seed_stability import ensure_dir, load_metrics, pick_device, train_one_seed

_ROOT = Path(__file__).resolve().parents[1]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate how much per-step variance can be explained by seed dependence for L-A-L-L-A-R."
    )
    parser.add_argument("--architecture_id", type=int, default=196)
    parser.add_argument("--architecture_code", type=str, default="L-A-L-L-A-R")
    parser.add_argument("--short_seed_count", type=int, default=16)
    parser.add_argument("--seed_selector", type=int, default=20260609)
    parser.add_argument("--seed_pool_max", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.005)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--nt", type=int, default=501)
    parser.add_argument("--cfg_seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--val_base_seed", type=int, default=1000000)
    parser.add_argument("--val_step", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=50)
    parser.add_argument("--data_mode", type=str, default="slices", choices=("full", "slices"))
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--meta_dataset_csv",
        type=str,
        default=str(
            _ROOT
            / "outputs"
            / "toy_diffusion"
            / "meta_model_family"
            / "server_v3_full328_residual1000_meta_dataset.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            _ROOT
            / "outputs"
            / "toy_diffusion"
            / "single_arch_seed_stability"
            / "lallar_v3_seed_variance_share"
        ),
    )
    parser.add_argument("--bootstrap_reps", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260609)
    parser.add_argument("--force", action="store_true")
    return parser


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def choose_short_seeds(selector: int, short_seed_count: int, seed_pool_max: int) -> list[int]:
    if short_seed_count < 2:
        raise ValueError("short_seed_count must be >= 2 to estimate a variance")
    rng = random.Random(selector)
    return rng.sample(range(1, seed_pool_max + 1), short_seed_count)


def load_full_rows(meta_csv: Path) -> list[dict[str, object]]:
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
                    "val_loss": float(row["val_loss"]),
                }
            )
    if not rows:
        raise ValueError(f"no usable rows found in {meta_csv}")
    return rows


def build_full_matrix(rows: list[dict[str, object]], expected_steps: list[int]) -> np.ndarray:
    by_arch: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_arch[int(row["architecture_id"])][int(row["step"])] = float(row["val_loss"])
    matrix = np.full((len(by_arch), len(expected_steps)), np.nan, dtype=np.float64)
    for row_idx, arch_id in enumerate(sorted(by_arch)):
        for col_idx, step in enumerate(expected_steps):
            value = by_arch[arch_id].get(step)
            if value is None:
                raise ValueError(f"full dataset missing step={step} for architecture_id={arch_id}")
            matrix[row_idx, col_idx] = value
    return matrix


def available_full_steps(rows: list[dict[str, object]]) -> list[int]:
    return sorted({int(row["step"]) for row in rows})


def build_seed_matrix(metrics_paths: list[Path]) -> tuple[list[int], np.ndarray]:
    step_maps: list[dict[int, float]] = []
    for path in metrics_paths:
        rows = load_metrics(path)
        step_map = {int(row["step"]): float(row["val_loss"]) for row in rows}
        step_maps.append(step_map)

    if not step_maps:
        raise ValueError("no seed metrics found")
    common_steps = sorted(set.intersection(*(set(step_map.keys()) for step_map in step_maps)))
    if not common_steps:
        raise ValueError("seed runs have no common evaluation steps")

    matrix = np.full((len(step_maps), len(common_steps)), np.nan, dtype=np.float64)
    for row_idx, step_map in enumerate(step_maps):
        for col_idx, step in enumerate(common_steps):
            matrix[row_idx, col_idx] = step_map[step]
    return common_steps, matrix


def subset_seed_matrix(
    steps: list[int],
    matrix: np.ndarray,
    kept_steps: list[int],
) -> tuple[list[int], np.ndarray]:
    keep_lookup = {step: idx for idx, step in enumerate(steps)}
    cols = [keep_lookup[step] for step in kept_steps]
    return kept_steps, matrix[:, cols]


def bootstrap_variance_statistics(
    seed_matrix: np.ndarray,
    full_matrix: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_seed = seed_matrix.shape[0]
    n_full = full_matrix.shape[0]
    if n_seed < 2 or n_full < 2:
        raise ValueError("need at least two rows in both seed_matrix and full_matrix")

    seed_var_boot = np.empty((reps, seed_matrix.shape[1]), dtype=np.float64)
    full_var_boot = np.empty((reps, full_matrix.shape[1]), dtype=np.float64)
    ratio_boot = np.empty((reps, seed_matrix.shape[1]), dtype=np.float64)

    for rep in range(reps):
        seed_idx = rng.integers(0, n_seed, size=n_seed)
        full_idx = rng.integers(0, n_full, size=n_full)
        seed_var = np.var(seed_matrix[seed_idx, :], axis=0, ddof=1)
        full_var = np.var(full_matrix[full_idx, :], axis=0, ddof=1)
        seed_var_boot[rep, :] = seed_var
        full_var_boot[rep, :] = full_var
        ratio_boot[rep, :] = seed_var / np.maximum(full_var, 1e-30)

    return {
        "seed_var_lo": np.percentile(seed_var_boot, 2.5, axis=0),
        "seed_var_hi": np.percentile(seed_var_boot, 97.5, axis=0),
        "full_var_lo": np.percentile(full_var_boot, 2.5, axis=0),
        "full_var_hi": np.percentile(full_var_boot, 97.5, axis=0),
        "ratio_lo": np.percentile(ratio_boot, 2.5, axis=0),
        "ratio_hi": np.percentile(ratio_boot, 97.5, axis=0),
    }


def summarize_ratio_windows(steps: np.ndarray, ratio: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, dict[str, float]]:
    windows = {
        "early_0_100": (steps >= 0) & (steps <= 100),
        "mid_150_500": (steps >= 150) & (steps <= 500),
        "late_550_999": (steps >= 550) & (steps <= 999),
        "all_steps": np.ones_like(steps, dtype=bool),
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in windows.items():
        if not np.any(mask):
            out[name] = {
                "mean_ratio": float("nan"),
                "median_ratio": float("nan"),
                "mean_ratio_ci_lo": float("nan"),
                "mean_ratio_ci_hi": float("nan"),
            }
            continue
        out[name] = {
            "mean_ratio": float(np.mean(ratio[mask])),
            "median_ratio": float(np.median(ratio[mask])),
            "mean_ratio_ci_lo": float(np.mean(lo[mask])),
            "mean_ratio_ci_hi": float(np.mean(hi[mask])),
        }
    return out


def build_variance_rows(
    steps: list[int],
    seed_var: np.ndarray,
    full_var: np.ndarray,
    ci: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    ratio = seed_var / np.maximum(full_var, 1e-30)
    for idx, step in enumerate(steps):
        rows.append(
            {
                "step": float(step),
                "seed_variance": float(seed_var[idx]),
                "full_variance": float(full_var[idx]),
                "variance_share": float(ratio[idx]),
                "seed_variance_ci_lo": float(ci["seed_var_lo"][idx]),
                "seed_variance_ci_hi": float(ci["seed_var_hi"][idx]),
                "full_variance_ci_lo": float(ci["full_var_lo"][idx]),
                "full_variance_ci_hi": float(ci["full_var_hi"][idx]),
                "variance_share_ci_lo": float(ci["ratio_lo"][idx]),
                "variance_share_ci_hi": float(ci["ratio_hi"][idx]),
            }
        )
    return rows


def make_curves_plot(out_dir: Path, steps: np.ndarray, seed_matrix: np.ndarray) -> Path:
    plt.figure(figsize=(10.0, 6.0))
    x = steps + 1.0
    for row in seed_matrix:
        plt.plot(x, row, color="#1f77b4", alpha=0.22, linewidth=1.1)
    median = np.median(seed_matrix, axis=0)
    q25 = np.percentile(seed_matrix, 25, axis=0)
    q75 = np.percentile(seed_matrix, 75, axis=0)
    plt.plot(x, median, color="#0b4fa2", linewidth=2.4, label="16-seed median")
    plt.fill_between(x, q25, q75, color="#5fa8ff", alpha=0.25, label="25-75%")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("training step + 1")
    plt.ylabel("validation loss")
    plt.title("L-A-L-L-A-R seed trajectories (16 runs, 1000 steps)")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "lallar_16_seed_curves_loglog.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def make_variance_plot(
    out_dir: Path,
    steps: np.ndarray,
    seed_var: np.ndarray,
    full_var: np.ndarray,
    ci: dict[str, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True, height_ratios=(1.3, 1.0))
    x = steps

    ax = axes[0]
    ax.plot(x, full_var, color="#1f77b4", linewidth=2.2, label="full328 cross-architecture variance")
    ax.fill_between(x, ci["full_var_lo"], ci["full_var_hi"], color="#1f77b4", alpha=0.16)
    ax.plot(x, seed_var, color="#d95f02", linewidth=2.0, label="L-A-L-L-A-R seed variance")
    ax.fill_between(x, ci["seed_var_lo"], ci["seed_var_hi"], color="#d95f02", alpha=0.18)
    ax.set_yscale("log")
    ax.set_ylabel("variance of val loss")
    ax.set_title("How much of per-step variance can seed dependence explain?")
    ax.legend()
    ax.grid(alpha=0.25)

    ratio = seed_var / np.maximum(full_var, 1e-30)
    ax = axes[1]
    ax.plot(x, ratio, color="#7f3c8d", linewidth=2.2, label="seed variance / full variance")
    ax.fill_between(x, ci["ratio_lo"], ci["ratio_hi"], color="#7f3c8d", alpha=0.2, label="95% bootstrap CI")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ymax = max(1.05, float(np.max(ci["ratio_hi"]) * 1.08))
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("training step")
    ax.set_ylabel("variance share")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "lallar_seed_variance_share_vs_full328.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def make_log_variance_plot(
    out_dir: Path,
    steps: np.ndarray,
    seed_log_var: np.ndarray,
    full_log_var: np.ndarray,
    ci: dict[str, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True, height_ratios=(1.3, 1.0))
    x = steps

    ax = axes[0]
    ax.plot(x, full_log_var, color="#1f77b4", linewidth=2.2, label="full328 cross-architecture log-variance")
    ax.fill_between(x, ci["full_var_lo"], ci["full_var_hi"], color="#1f77b4", alpha=0.16)
    ax.plot(x, seed_log_var, color="#d95f02", linewidth=2.0, label="L-A-L-L-A-R seed log-variance")
    ax.fill_between(x, ci["seed_var_lo"], ci["seed_var_hi"], color="#d95f02", alpha=0.18)
    ax.set_yscale("log")
    ax.set_ylabel("variance of log10(val loss)")
    ax.set_title("Seed dependence vs full328 variance on log10(loss) scale")
    ax.legend()
    ax.grid(alpha=0.25)

    ratio = seed_log_var / np.maximum(full_log_var, 1e-30)
    ax = axes[1]
    ax.plot(x, ratio, color="#7f3c8d", linewidth=2.2, label="seed log-variance / full log-variance")
    ax.fill_between(x, ci["ratio_lo"], ci["ratio_hi"], color="#7f3c8d", alpha=0.2, label="95% bootstrap CI")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ymax = max(1.05, float(np.max(ci["ratio_hi"]) * 1.08))
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("training step")
    ax.set_ylabel("variance share")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "lallar_log10_seed_variance_share_vs_full328.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = ensure_dir(args.output_dir)
    out_dir = Path(out_dir)

    from meta_model_train.diffusion_dataset import Diffusion2DConfig

    short_seeds = choose_short_seeds(args.seed_selector, args.short_seed_count, args.seed_pool_max)
    dump_json(
        out_dir / "seed_manifest.json",
        {
            "architecture_code": args.architecture_code,
            "architecture_id": args.architecture_id,
            "seed_selector": args.seed_selector,
            "short_seed_count": args.short_seed_count,
            "short_seeds": short_seeds,
            "settings": {
                "hidden_dim": args.hidden_dim,
                "batch_size": args.batch_size,
                "val_batch_size": args.val_batch_size,
                "max_steps": args.max_steps,
                "val_every": args.val_every,
            },
        },
    )

    diff_cfg = Diffusion2DConfig(
        nx=args.nx,
        ny=args.ny,
        L=args.L,
        D=args.D,
        T=args.T,
        nt=args.nt,
        seed=args.cfg_seed,
    )
    device = pick_device(args.device)

    run_summaries: list[dict[str, object]] = []
    for idx, seed in enumerate(short_seeds, start=1):
        label = f"seed_{idx:02d}_{seed}"
        summary = train_one_seed(
            args,
            torch_seed=seed,
            max_steps=args.max_steps,
            label=label,
            device=device,
            diff_cfg=diff_cfg,
            out_dir=out_dir,
        )
        run_summaries.append(summary)

    metrics_paths = [Path(str(summary["metrics_csv"])) for summary in run_summaries]
    full_rows = load_full_rows(Path(args.meta_dataset_csv))
    steps, seed_matrix = build_seed_matrix(metrics_paths)
    common_steps = sorted(set(steps).intersection(available_full_steps(full_rows)))
    if not common_steps:
        raise ValueError("seed runs and full dataset have no shared evaluation steps")
    if len(common_steps) != len(steps):
        steps, seed_matrix = subset_seed_matrix(steps, seed_matrix, common_steps)
    full_matrix = build_full_matrix(full_rows, steps)

    seed_var = np.var(seed_matrix, axis=0, ddof=1)
    full_var = np.var(full_matrix, axis=0, ddof=1)
    ratio = seed_var / np.maximum(full_var, 1e-30)
    seed_log_matrix = np.log10(np.maximum(seed_matrix, 1e-30))
    full_log_matrix = np.log10(np.maximum(full_matrix, 1e-30))
    seed_log_var = np.var(seed_log_matrix, axis=0, ddof=1)
    full_log_var = np.var(full_log_matrix, axis=0, ddof=1)
    log_ratio = seed_log_var / np.maximum(full_log_var, 1e-30)

    rng = np.random.default_rng(args.bootstrap_seed)
    ci = bootstrap_variance_statistics(seed_matrix, full_matrix, args.bootstrap_reps, rng)
    log_ci = bootstrap_variance_statistics(seed_log_matrix, full_log_matrix, args.bootstrap_reps, rng)
    curves_plot = make_curves_plot(out_dir, np.asarray(steps, dtype=np.int64), seed_matrix)
    variance_plot = make_variance_plot(out_dir, np.asarray(steps, dtype=np.int64), seed_var, full_var, ci)
    log_variance_plot = make_log_variance_plot(
        out_dir,
        np.asarray(steps, dtype=np.int64),
        seed_log_var,
        full_log_var,
        log_ci,
    )
    ratio_windows = summarize_ratio_windows(
        np.asarray(steps, dtype=np.int64),
        ratio,
        ci["ratio_lo"],
        ci["ratio_hi"],
    )
    log_ratio_windows = summarize_ratio_windows(
        np.asarray(steps, dtype=np.int64),
        log_ratio,
        log_ci["ratio_lo"],
        log_ci["ratio_hi"],
    )

    per_step_rows = build_variance_rows(steps, seed_var, full_var, ci)
    log_per_step_rows = build_variance_rows(steps, seed_log_var, full_log_var, log_ci)

    summary = {
        "architecture_code": args.architecture_code,
        "architecture_id": args.architecture_id,
        "device_resolved": str(device),
        "meta_dataset_csv": str(Path(args.meta_dataset_csv).resolve()),
        "num_seed_runs": len(run_summaries),
        "num_full_architectures": int(full_matrix.shape[0]),
        "seed_manifest": str((out_dir / "seed_manifest.json").resolve()),
        "seed_metrics_csvs": [str(path.resolve()) for path in metrics_paths],
        "plots": {
            "seed_curves_plot": str(curves_plot.resolve()),
            "variance_share_plot": str(variance_plot.resolve()),
            "log_variance_share_plot": str(log_variance_plot.resolve()),
        },
        "ratio_windows": ratio_windows,
        "log_ratio_windows": log_ratio_windows,
        "per_step_rows": per_step_rows,
        "log_per_step_rows": log_per_step_rows,
        "run_summaries": run_summaries,
    }
    dump_json(out_dir / "summary.json", summary)
    print(f"[analyze_lallar_seed_variance_share] output_dir={out_dir}")
    print(f"[analyze_lallar_seed_variance_share] variance_plot={variance_plot}")


if __name__ == "__main__":
    main()
