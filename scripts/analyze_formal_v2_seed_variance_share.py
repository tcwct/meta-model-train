from __future__ import annotations

import argparse
import csv
import concurrent.futures
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_model_train.diffusion_dataset import Diffusion2DConfig
from run_single_arch_seed_scaling import ensure_dir, load_metrics, pick_device, train_one_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate how much 2.0-family per-step variance can be explained by seed dependence."
    )
    parser.add_argument(
        "--meta_dataset_csv",
        type=str,
        default=str(_ROOT / "outputs" / "formal_v2_64" / "merged" / "meta_dataset.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_ROOT / "outputs" / "formal_v2_64" / "seed_variance_share"),
    )
    parser.add_argument(
        "--architecture",
        action="append",
        default=[],
        help="Architecture spec as label:architecture_id:architecture_code. Can be passed multiple times.",
    )
    parser.add_argument("--short_seed_count", type=int, default=16)
    parser.add_argument("--seed_selector", type=int, default=20260609)
    parser.add_argument("--seed_pool_max", type=int, default=10000)
    parser.add_argument("--bootstrap_reps", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260609)
    parser.add_argument("--num_workers", type=int, default=1)
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
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--val_every", type=int, default=50)
    parser.add_argument("--data_mode", type=str, default="slices", choices=("full", "slices"))
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--force", action="store_true")
    return parser


def default_architecture_specs() -> list[dict[str, object]]:
    return [
        {
            "label": "best_balanced_4A4M",
            "architecture_id": 31,
            "architecture_code": "A-M-M-A-M-A-A-M",
            "rationale": "Best architecture in the current 64-sample run; represents strong balanced mixed designs.",
        },
        {
            "label": "strong_mlp_heavy_2A6M",
            "architecture_id": 58,
            "architecture_code": "M-M-M-A-A-M-M-M",
            "rationale": "One of the strongest low-attention models; represents the MLP-heavy competitive regime.",
        },
    ]


def parse_architecture_specs(raw_specs: list[str]) -> list[dict[str, object]]:
    if not raw_specs:
        return default_architecture_specs()
    parsed: list[dict[str, object]] = []
    for spec in raw_specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"invalid --architecture spec: {spec!r}; expected label:architecture_id:architecture_code")
        label, architecture_id, architecture_code = parts
        parsed.append(
            {
                "label": label,
                "architecture_id": int(architecture_id),
                "architecture_code": architecture_code,
                "rationale": "user provided",
            }
        )
    return parsed


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def choose_short_seeds(selector: int, short_seed_count: int, seed_pool_max: int) -> list[int]:
    if short_seed_count < 2:
        raise SystemExit("short_seed_count must be >= 2")
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
        raise SystemExit(f"no usable rows found in {meta_csv}")
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
                raise SystemExit(f"full dataset missing step={step} for architecture_id={arch_id}")
            matrix[row_idx, col_idx] = value
    return matrix


def available_full_steps(rows: list[dict[str, object]]) -> list[int]:
    return sorted({int(row["step"]) for row in rows})


def build_seed_matrix(metrics_paths: list[Path]) -> tuple[list[int], np.ndarray]:
    step_maps: list[dict[int, float]] = []
    for path in metrics_paths:
        rows = load_metrics(path)
        step_map = {int(row["step"]): float(row["val_loss"]) for row in rows if not math.isnan(float(row["val_loss"]))}
        step_maps.append(step_map)

    if not step_maps:
        raise SystemExit("no seed metrics found")
    common_steps = sorted(set.intersection(*(set(step_map.keys()) for step_map in step_maps)))
    if not common_steps:
        raise SystemExit("seed runs have no common evaluation steps")

    matrix = np.full((len(step_maps), len(common_steps)), np.nan, dtype=np.float64)
    for row_idx, step_map in enumerate(step_maps):
        for col_idx, step in enumerate(common_steps):
            matrix[row_idx, col_idx] = step_map[step]
    return common_steps, matrix


def subset_seed_matrix(steps: list[int], matrix: np.ndarray, kept_steps: list[int]) -> tuple[list[int], np.ndarray]:
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
    seed_var_boot = np.empty((reps, seed_matrix.shape[1]), dtype=np.float64)
    full_var_boot = np.empty((reps, full_matrix.shape[1]), dtype=np.float64)
    ratio_boot = np.empty((reps, full_matrix.shape[1]), dtype=np.float64)

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
    ratio = seed_var / np.maximum(full_var, 1e-30)
    out: list[dict[str, float]] = []
    for idx, step in enumerate(steps):
        out.append(
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
    return out


def serialize_ci(ci: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: value.tolist() for key, value in ci.items()}


def make_seed_curves_plot(
    output_path: Path,
    results: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.6 * len(results), 5.2), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        steps = np.asarray(result["steps"], dtype=np.float64) + 1.0
        seed_matrix = np.asarray(result["seed_matrix"], dtype=np.float64)
        for row in seed_matrix:
            ax.plot(steps, row, color="#1f77b4", alpha=0.20, linewidth=1.0)
        median = np.median(seed_matrix, axis=0)
        q25 = np.percentile(seed_matrix, 25, axis=0)
        q75 = np.percentile(seed_matrix, 75, axis=0)
        ax.plot(steps, median, color="#0b4fa2", linewidth=2.4, label="seed median")
        ax.fill_between(steps, q25, q75, color="#5fa8ff", alpha=0.25, label="25-75%")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(str(result["architecture_code"]))
        ax.set_xlabel("training step + 1")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("validation loss")
    axes[0].legend(frameon=False)
    fig.suptitle("Seed trajectories for two representative 2.0 architectures")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_variance_figure(
    output_path: Path,
    results: list[dict[str, object]],
    *,
    title: str,
    top_ylabel: str,
    bottom_ylabel: str,
    top_key: str,
    bottom_key: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        len(results),
        figsize=(6.6 * len(results), 8.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    if len(results) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)

    for col, result in enumerate(results):
        steps = np.asarray(result["steps"], dtype=np.float64)
        top = np.asarray(result[top_key], dtype=np.float64)
        bottom = np.asarray(result[bottom_key], dtype=np.float64)
        top_ci = result[f"{top_key}_ci"]
        bottom_ci = result[f"{bottom_key}_ci"]

        ax = axes[0, col]
        ax.plot(steps, top[:, 1], color="#1f77b4", linewidth=2.2, label="full-dataset variance")
        ax.fill_between(steps, top_ci["full_var_lo"], top_ci["full_var_hi"], color="#1f77b4", alpha=0.16)
        ax.plot(steps, top[:, 0], color="#d95f02", linewidth=2.0, label="seed variance")
        ax.fill_between(steps, top_ci["seed_var_lo"], top_ci["seed_var_hi"], color="#d95f02", alpha=0.18)
        ax.set_yscale("log")
        ax.set_title(str(result["architecture_code"]))
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel(top_ylabel)
            ax.legend(frameon=False)

        ax = axes[1, col]
        ax.plot(steps, bottom, color="#7f3c8d", linewidth=2.2, label="seed variance / full variance")
        ax.fill_between(steps, bottom_ci["ratio_lo"], bottom_ci["ratio_hi"], color="#7f3c8d", alpha=0.22)
        for y in (0.1, 0.25, 0.5, 1.0):
            ax.axhline(y, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ymax = max(1.05, float(np.max(bottom_ci["ratio_hi"]) * 1.08))
        ax.set_ylim(0.0, ymax)
        ax.grid(alpha=0.25)
        ax.set_xlabel("training step")
        if col == 0:
            ax.set_ylabel(bottom_ylabel)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_architecture_seed_sweep(
    args: argparse.Namespace,
    arch_spec: dict[str, object],
    short_seeds: list[int],
    output_dir: Path,
) -> list[Path]:
    run_root = ensure_dir(output_dir / str(arch_spec["label"]))
    jobs = []
    for idx, torch_seed in enumerate(short_seeds, start=1):
        jobs.append(
            {
                "args_dict": vars(args).copy(),
                "architecture_id": int(arch_spec["architecture_id"]),
                "architecture_code": str(arch_spec["architecture_code"]),
                "torch_seed": int(torch_seed),
                "label": f"seed_{idx:02d}_{torch_seed}",
                "run_root": str(run_root),
            }
        )

    if args.num_workers <= 1:
        return [Path(train_one_seed_worker(job)) for job in jobs]

    metrics_paths: list[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for metrics_csv in executor.map(train_one_seed_worker, jobs):
            metrics_paths.append(Path(metrics_csv))
    return metrics_paths


def train_one_seed_worker(job: dict[str, object]) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    args_dict = dict(job["args_dict"])
    train_args = argparse.Namespace(**args_dict)
    train_args.architecture_id = int(job["architecture_id"])
    train_args.architecture_code = str(job["architecture_code"])
    train_args.output_dir = str(job["run_root"])

    device = pick_device(train_args.device)
    diff_cfg = Diffusion2DConfig(
        nx=train_args.nx,
        ny=train_args.ny,
        L=train_args.L,
        D=train_args.D,
        T=train_args.T,
        nt=train_args.nt,
        seed=train_args.cfg_seed,
    )
    summary = train_one_seed(
        train_args,
        torch_seed=int(job["torch_seed"]),
        max_steps=int(train_args.max_steps),
        label=str(job["label"]),
        device=device,
        diff_cfg=diff_cfg,
        out_dir=Path(str(job["run_root"])),
    )
    return str(summary["metrics_csv"])


def compute_result_for_architecture(
    arch_spec: dict[str, object],
    metrics_paths: list[Path],
    full_steps: list[int],
    full_matrix: np.ndarray,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    seed_steps, seed_matrix = build_seed_matrix(metrics_paths)
    kept_steps = [step for step in full_steps if step in seed_steps]
    kept_steps, seed_matrix = subset_seed_matrix(seed_steps, seed_matrix, kept_steps)

    keep_lookup = {step: idx for idx, step in enumerate(full_steps)}
    full_subset = full_matrix[:, [keep_lookup[step] for step in kept_steps]]

    seed_var = np.var(seed_matrix, axis=0, ddof=1)
    full_var = np.var(full_subset, axis=0, ddof=1)
    raw_ci = bootstrap_variance_statistics(
        seed_matrix=seed_matrix,
        full_matrix=full_subset,
        reps=bootstrap_reps,
        rng=np.random.default_rng(bootstrap_seed + int(arch_spec["architecture_id"])),
    )

    log_seed_matrix = np.log10(np.maximum(seed_matrix, 1e-30))
    log_full_matrix = np.log10(np.maximum(full_subset, 1e-30))
    log_seed_var = np.var(log_seed_matrix, axis=0, ddof=1)
    log_full_var = np.var(log_full_matrix, axis=0, ddof=1)
    log_ci = bootstrap_variance_statistics(
        seed_matrix=log_seed_matrix,
        full_matrix=log_full_matrix,
        reps=bootstrap_reps,
        rng=np.random.default_rng(bootstrap_seed + 10000 + int(arch_spec["architecture_id"])),
    )

    final_losses = seed_matrix[:, -1]
    return {
        "label": arch_spec["label"],
        "architecture_id": int(arch_spec["architecture_id"]),
        "architecture_code": str(arch_spec["architecture_code"]),
        "rationale": str(arch_spec["rationale"]),
        "steps": kept_steps,
        "seed_matrix": seed_matrix.tolist(),
        "raw_var_matrix": np.stack([seed_var, full_var], axis=1).tolist(),
        "raw_ratio": (seed_var / np.maximum(full_var, 1e-30)).tolist(),
        "raw_var_matrix_ci": serialize_ci(raw_ci),
        "raw_ratio_ci": {"ratio_lo": raw_ci["ratio_lo"].tolist(), "ratio_hi": raw_ci["ratio_hi"].tolist()},
        "log_var_matrix": np.stack([log_seed_var, log_full_var], axis=1).tolist(),
        "log_ratio": (log_seed_var / np.maximum(log_full_var, 1e-30)).tolist(),
        "log_var_matrix_ci": serialize_ci(log_ci),
        "log_ratio_ci": {"ratio_lo": log_ci["ratio_lo"].tolist(), "ratio_hi": log_ci["ratio_hi"].tolist()},
        "raw_step_rows": build_variance_rows(kept_steps, seed_var, full_var, raw_ci),
        "log_step_rows": build_variance_rows(kept_steps, log_seed_var, log_full_var, log_ci),
        "raw_window_summary": summarize_ratio_windows(
            np.asarray(kept_steps), seed_var / np.maximum(full_var, 1e-30), raw_ci["ratio_lo"], raw_ci["ratio_hi"]
        ),
        "log_window_summary": summarize_ratio_windows(
            np.asarray(kept_steps),
            log_seed_var / np.maximum(log_full_var, 1e-30),
            log_ci["ratio_lo"],
            log_ci["ratio_hi"],
        ),
        "final_val_summary": {
            "min": float(np.min(final_losses)),
            "median": float(np.median(final_losses)),
            "max": float(np.max(final_losses)),
            "mean": float(np.mean(final_losses)),
        },
    }


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    meta_csv = Path(args.meta_dataset_csv).resolve()
    arch_specs = parse_architecture_specs(args.architecture)
    short_seeds = choose_short_seeds(args.seed_selector, args.short_seed_count, args.seed_pool_max)

    full_rows = load_full_rows(meta_csv)
    full_steps = available_full_steps(full_rows)
    full_matrix = build_full_matrix(full_rows, full_steps)

    results: list[dict[str, object]] = []
    for arch_spec in arch_specs:
        metrics_paths = run_architecture_seed_sweep(args, arch_spec, short_seeds, output_dir)
        result = compute_result_for_architecture(
            arch_spec=arch_spec,
            metrics_paths=metrics_paths,
            full_steps=full_steps,
            full_matrix=full_matrix,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
        )
        results.append(result)

    make_seed_curves_plot(output_dir / "two_arch_seed_curves_loglog.png", results)
    make_variance_figure(
        output_dir / "two_arch_seed_variance_share_raw.png",
        results,
        title="Seed variance vs formal_v2_64 cross-architecture variance",
        top_ylabel="variance of val loss",
        bottom_ylabel="seed variance / full variance",
        top_key="raw_var_matrix",
        bottom_key="raw_ratio",
    )
    make_variance_figure(
        output_dir / "two_arch_seed_variance_share_log10.png",
        results,
        title="Seed variance vs formal_v2_64 cross-architecture variance on log10(loss)",
        top_ylabel="variance of log10(val loss)",
        bottom_ylabel="seed variance / full variance",
        top_key="log_var_matrix",
        bottom_key="log_ratio",
    )

    summary = {
        "meta_dataset_csv": str(meta_csv),
        "output_dir": str(output_dir),
        "short_seeds": short_seeds,
        "architectures": arch_specs,
        "results": results,
    }
    dump_json(output_dir / "summary.json", summary)
    print(f"[analyze_formal_v2_seed_variance_share] wrote_dir={output_dir}")


if __name__ == "__main__":
    main()
