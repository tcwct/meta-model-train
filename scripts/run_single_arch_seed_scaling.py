from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_model_train.diffusion_dataset import Diffusion2DConfig, generate_single_step_batch, sample_u0_seeds_batch
from meta_model_train.minimal_arch_model import (
    MinimalArchConfig,
    MinimalArchModel,
    architecture_token_counts,
    canonicalize_architecture_code,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run short seed sweeps for one architecture, then continue the median run to a longer horizon."
    )
    parser.add_argument("--architecture_id", type=int, default=3)
    parser.add_argument("--architecture_code", type=str, default="M-M-M-M-M-M-A-A")
    parser.add_argument("--seed_selector", type=int, default=20260609)
    parser.add_argument("--short_seed_count", type=int, default=4)
    parser.add_argument("--seed_pool_max", type=int, default=999)
    parser.add_argument("--short_steps", type=int, default=1000)
    parser.add_argument("--long_steps", type=int, default=4000)
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
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--data_mode", type=str, default="slices", choices=("full", "slices"))
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "toy_diffusion", "single_arch_seed_scaling", "m6a2_default"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[run_single_arch_seed_scaling] cuda not available, using cpu")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def evaluate_fixed_batch(
    model: MinimalArchModel,
    cfg: Diffusion2DConfig,
    k: int,
    seeds,
    device: torch.device,
    *,
    data_mode: str,
) -> float:
    model.eval()
    with torch.no_grad():
        inputs_np, target_np = generate_single_step_batch(cfg, k, seeds, data_mode=data_mode)
        x = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
        pred = model(x)
        return float(F.mse_loss(pred, y).item())


def append_metrics_row(
    csv_path: Path,
    *,
    step: int,
    train_loss: float | None,
    val_loss: float | None,
    elapsed_s: float,
) -> None:
    fieldnames = ("step", "train_loss", "val_loss", "elapsed_s")
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "step": step,
                "train_loss": "" if train_loss is None else f"{train_loss:.10f}",
                "val_loss": "" if val_loss is None else f"{val_loss:.10f}",
                "elapsed_s": f"{elapsed_s:.3f}",
            }
        )


def load_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "step": float(row["step"]),
                "elapsed_s": float(row["elapsed_s"]),
            }
            train_loss = row["train_loss"].strip()
            val_loss = row["val_loss"].strip()
            parsed["train_loss"] = math.nan if not train_loss else float(train_loss)
            parsed["val_loss"] = math.nan if not val_loss else float(val_loss)
            rows.append(parsed)
    return rows


def save_checkpoint(
    path: Path,
    *,
    model: MinimalArchModel,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    best_val_loss: float,
    best_step: int,
    final_train_loss: float,
    final_val_loss: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_steps": int(completed_steps),
        "best_val_loss": float(best_val_loss),
        "best_step": int(best_step),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
    }
    torch.save(payload, path)


def choose_short_seeds(selector: int, short_seed_count: int, seed_pool_max: int) -> list[int]:
    if short_seed_count < 1:
        raise ValueError("short_seed_count must be >= 1")
    rng = random.Random(selector)
    return rng.sample(range(1, seed_pool_max + 1), short_seed_count)


def build_model(args: argparse.Namespace, device: torch.device) -> MinimalArchModel:
    model_cfg = MinimalArchConfig(
        image_size=args.nx,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        architecture_code=args.architecture_code,
    )
    return MinimalArchModel(model_cfg).to(device)


def train_one_seed(
    args: argparse.Namespace,
    *,
    torch_seed: int,
    max_steps: int,
    label: str,
    device: torch.device,
    diff_cfg: Diffusion2DConfig,
    out_dir: Path,
) -> dict[str, object]:
    run_dir = ensure_dir(out_dir / label)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"

    if summary_path.exists() and checkpoint_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "torch_seed": torch_seed,
            "max_steps": max_steps,
            "label": label,
            "architecture_code": model.architecture_code,
            "architecture_id": args.architecture_id,
            "device_resolved": str(device),
            "parameter_count": count_parameters(model),
        }
    )
    config_payload.update(architecture_token_counts(model.architecture_code))
    dump_json(config_path, config_payload)

    if metrics_path.exists():
        metrics_path.unlink()
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    val_seeds = sample_u0_seeds_batch(args.val_base_seed, args.val_step, args.val_batch_size)
    init_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
    append_metrics_row(metrics_path, step=-1, train_loss=None, val_loss=init_val, elapsed_s=0.0)
    print(f"[run_single_arch_seed_scaling] {label} step=-1 val_loss={init_val:.6f}")

    best_val = init_val
    best_step = -1
    final_train = math.nan
    final_val = init_val
    t0 = time.perf_counter()

    for step in range(max_steps):
        model.train()
        seeds = sample_u0_seeds_batch(args.base_seed, step, args.batch_size)
        inputs_np, target_np = generate_single_step_batch(diff_cfg, args.k, seeds, data_mode=args.data_mode)
        x = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        final_train = float(loss.item())
        if step % args.log_every == 0 or step == max_steps - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"[run_single_arch_seed_scaling] {label} step={step} "
                f"train_loss={final_train:.6f} elapsed_s={elapsed:.1f}"
            )

        if step % args.val_every == 0 or step == max_steps - 1:
            elapsed = time.perf_counter() - t0
            final_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
            append_metrics_row(metrics_path, step=step, train_loss=final_train, val_loss=final_val, elapsed_s=elapsed)
            if final_val < best_val:
                best_val = final_val
                best_step = step
            print(
                f"[run_single_arch_seed_scaling] {label} step={step} "
                f"val_loss={final_val:.6f} elapsed_s={elapsed:.1f}"
            )

    runtime_s = time.perf_counter() - t0
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        completed_steps=max_steps,
        best_val_loss=best_val,
        best_step=best_step,
        final_train_loss=final_train,
        final_val_loss=final_val,
    )
    summary = {
        "label": label,
        "torch_seed": torch_seed,
        "max_steps": max_steps,
        "initial_val_loss": init_val,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "best_step": best_step,
        "runtime_s": runtime_s,
        "metrics_csv": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
    }
    dump_json(summary_path, summary)
    return summary


def select_median_run(short_runs: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(short_runs, key=lambda row: (float(row["final_val_loss"]), int(row["torch_seed"])))
    n = len(ordered)
    if n % 2 == 1:
        selected = ordered[n // 2]
    else:
        lower = ordered[n // 2 - 1]
        upper = ordered[n // 2]
        target = 0.5 * (float(lower["final_val_loss"]) + float(upper["final_val_loss"]))
        candidates = [lower, upper]
        selected = min(
            candidates,
            key=lambda row: (
                abs(float(row["final_val_loss"]) - target),
                float(row["final_val_loss"]),
                int(row["torch_seed"]),
            ),
        )
    return selected


def continue_run_to_long_horizon(
    args: argparse.Namespace,
    *,
    short_summary: dict[str, object],
    device: torch.device,
    diff_cfg: Diffusion2DConfig,
    out_dir: Path,
) -> dict[str, object]:
    torch_seed = int(short_summary["torch_seed"])
    source_metrics = Path(str(short_summary["metrics_csv"]))
    source_checkpoint = Path(str(short_summary["checkpoint_path"]))
    label = f"median_seed_{torch_seed}_continued_to_{args.long_steps}"
    run_dir = ensure_dir(out_dir / label)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"

    if summary_path.exists() and checkpoint_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    resume_payload = torch.load(source_checkpoint, map_location=device)
    model.load_state_dict(resume_payload["model_state"])
    optimizer.load_state_dict(resume_payload["optimizer_state"])
    completed_steps = int(resume_payload["completed_steps"])
    if completed_steps != args.short_steps:
        raise ValueError(
            f"expected completed_steps={args.short_steps} from short run, got {completed_steps}"
        )

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "torch_seed": torch_seed,
            "max_steps": args.long_steps,
            "label": label,
            "architecture_code": canonicalize_architecture_code(args.architecture_code),
            "architecture_id": args.architecture_id,
            "device_resolved": str(device),
            "parameter_count": count_parameters(model),
            "continued_from_label": short_summary["label"],
            "continued_from_metrics_csv": str(source_metrics),
            "continued_from_checkpoint_path": str(source_checkpoint),
        }
    )
    config_payload.update(architecture_token_counts(args.architecture_code))
    dump_json(config_path, config_payload)

    if metrics_path.exists():
        metrics_path.unlink()
    shutil.copy2(source_metrics, metrics_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    val_seeds = sample_u0_seeds_batch(args.val_base_seed, args.val_step, args.val_batch_size)
    best_val = float(resume_payload["best_val_loss"])
    best_step = int(resume_payload["best_step"])
    final_train = float(resume_payload["final_train_loss"])
    final_val = float(resume_payload["final_val_loss"])

    existing_rows = load_metrics(source_metrics)
    elapsed_offset = max((row["elapsed_s"] for row in existing_rows), default=0.0)
    t0 = time.perf_counter()
    for step in range(completed_steps, args.long_steps):
        model.train()
        seeds = sample_u0_seeds_batch(args.base_seed, step, args.batch_size)
        inputs_np, target_np = generate_single_step_batch(diff_cfg, args.k, seeds, data_mode=args.data_mode)
        x = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        final_train = float(loss.item())
        if step % args.log_every == 0 or step == args.long_steps - 1:
            elapsed = elapsed_offset + (time.perf_counter() - t0)
            print(
                f"[run_single_arch_seed_scaling] {label} step={step} "
                f"train_loss={final_train:.6f} elapsed_s={elapsed:.1f}"
            )

        if step % args.val_every == 0 or step == args.long_steps - 1:
            elapsed = elapsed_offset + (time.perf_counter() - t0)
            final_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
            append_metrics_row(metrics_path, step=step, train_loss=final_train, val_loss=final_val, elapsed_s=elapsed)
            if final_val < best_val:
                best_val = final_val
                best_step = step
            print(
                f"[run_single_arch_seed_scaling] {label} step={step} "
                f"val_loss={final_val:.6f} elapsed_s={elapsed:.1f}"
            )

    runtime_s = elapsed_offset + (time.perf_counter() - t0)
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        completed_steps=args.long_steps,
        best_val_loss=best_val,
        best_step=best_step,
        final_train_loss=final_train,
        final_val_loss=final_val,
    )
    summary = {
        "label": label,
        "torch_seed": torch_seed,
        "max_steps": args.long_steps,
        "initial_val_loss": float(short_summary["initial_val_loss"]),
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "best_step": best_step,
        "runtime_s": runtime_s,
        "metrics_csv": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
        "continued_from_label": short_summary["label"],
    }
    dump_json(summary_path, summary)
    return summary


def make_plot(
    out_dir: Path,
    short_runs: list[dict[str, object]],
    long_run: dict[str, object],
    architecture_code: str,
) -> Path:
    plt.figure(figsize=(9.5, 6.2))
    for run in short_runs:
        rows = [row for row in load_metrics(Path(str(run["metrics_csv"]))) if row["step"] >= 0 and not math.isnan(row["val_loss"])]
        x = [row["step"] + 1.0 for row in rows]
        y = [row["val_loss"] for row in rows]
        plt.plot(x, y, linewidth=1.6, alpha=0.78, label=f"short seed={run['torch_seed']}")

    long_rows = [row for row in load_metrics(Path(str(long_run["metrics_csv"]))) if row["step"] >= 0 and not math.isnan(row["val_loss"])]
    x_long = [row["step"] + 1.0 for row in long_rows]
    y_long = [row["val_loss"] for row in long_rows]
    plt.plot(
        x_long,
        y_long,
        linewidth=2.7,
        alpha=0.95,
        color="black",
        label=f"continued median seed={long_run['torch_seed']}",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Optimization Step + 1 (log scale)")
    plt.ylabel("Validation Loss (log scale)")
    plt.title(f"{architecture_code} Seed Dependence and Scaling")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "seed_scaling_loglog.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return plot_path


def write_ranked_short_summary(path: Path, short_runs: list[dict[str, object]]) -> None:
    ordered = sorted(short_runs, key=lambda row: float(row["final_val_loss"]))
    payload = []
    for rank, row in enumerate(ordered, start=1):
        payload.append(
            {
                "rank": rank,
                "torch_seed": int(row["torch_seed"]),
                "final_val_loss": float(row["final_val_loss"]),
                "best_val_loss": float(row["best_val_loss"]),
                "best_step": int(row["best_step"]),
                "label": row["label"],
            }
        )
    dump_json(path, {"ordered_by_final_val_loss": payload})


def main() -> None:
    args = build_argparser().parse_args()
    device = pick_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    short_seeds = choose_short_seeds(args.seed_selector, args.short_seed_count, args.seed_pool_max)
    seed_manifest = {
        "seed_selector": args.seed_selector,
        "short_seeds": short_seeds,
        "architecture_id": args.architecture_id,
        "architecture_code": canonicalize_architecture_code(args.architecture_code),
        "settings": {
            "short_steps": args.short_steps,
            "long_steps": args.long_steps,
            "val_every": args.val_every,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
        },
    }
    dump_json(out_dir / "seed_manifest.json", seed_manifest)

    diff_cfg = Diffusion2DConfig(
        nx=args.nx,
        ny=args.ny,
        L=args.L,
        D=args.D,
        T=args.T,
        nt=args.nt,
        seed=args.cfg_seed,
    )

    short_runs: list[dict[str, object]] = []
    for idx, seed in enumerate(short_seeds, start=1):
        label = f"short_seed_{idx}_{seed}"
        summary = train_one_seed(
            args,
            torch_seed=seed,
            max_steps=args.short_steps,
            label=label,
            device=device,
            diff_cfg=diff_cfg,
            out_dir=out_dir,
        )
        short_runs.append(summary)

    selected_short = select_median_run(short_runs)
    long_run = continue_run_to_long_horizon(
        args,
        short_summary=selected_short,
        device=device,
        diff_cfg=diff_cfg,
        out_dir=out_dir,
    )

    write_ranked_short_summary(out_dir / "short_run_ranking.json", short_runs)
    plot_path = make_plot(out_dir, short_runs, long_run, canonicalize_architecture_code(args.architecture_code))
    aggregate = {
        "architecture_id": args.architecture_id,
        "architecture_code": canonicalize_architecture_code(args.architecture_code),
        "device_resolved": str(device),
        "short_runs": short_runs,
        "selected_median_short_run": selected_short,
        "continued_long_run": long_run,
        "plot_path": str(plot_path),
    }
    dump_json(out_dir / "aggregate_summary.json", aggregate)
    print(f"[run_single_arch_seed_scaling] selected_median_seed={selected_short['torch_seed']}")
    print(f"[run_single_arch_seed_scaling] done output_dir={out_dir}")
    print(f"[run_single_arch_seed_scaling] plot={plot_path}")


if __name__ == "__main__":
    main()
