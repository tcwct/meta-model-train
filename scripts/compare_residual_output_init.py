from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    patchify_2d,
    unpatchify_2d,
    zero_init_linear,
)


class ResidualIOMinimalArchModel(MinimalArchModel):
    def __init__(self, cfg: MinimalArchConfig, *, zero_init_output: bool):
        super().__init__(cfg)
        if zero_init_output:
            zero_init_linear(self.output_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected input rank 4, got shape {tuple(x.shape)}")
        patches = patchify_2d(x, self.cfg.patch_size)
        hidden = self.input_proj(patches)
        for op in self.ops:
            hidden = op(hidden)
        delta_patches = self.output_proj(hidden)
        return unpatchify_2d(patches + delta_patches, self.cfg.patch_size)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare residual IO models with zero-initialized vs random-initialized output projection."
    )
    parser.add_argument("--architecture_id", type=int, default=3)
    parser.add_argument("--architecture_code", type=str, default="M-M-M-M-M-M-A-A")
    parser.add_argument("--seed_selector", type=int, default=20260609)
    parser.add_argument("--seed_count", type=int, default=4)
    parser.add_argument("--seed_pool_max", type=int, default=999)
    parser.add_argument("--residual_zero_seed", type=int, default=None)
    parser.add_argument("--residual_random_seed", type=int, default=None)
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
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--data_mode", type=str, default="slices", choices=("full", "slices"))
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("outputs", "toy_diffusion", "init_compare", "residual_output_init_m6a2"),
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
    print("[compare_residual_output_init] cuda not available, using cpu")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def choose_seeds(selector: int, seed_count: int, seed_pool_max: int) -> list[int]:
    if seed_count < 1:
        raise ValueError("seed_count must be >= 1")
    rng = random.Random(selector)
    return rng.sample(range(1, seed_pool_max + 1), seed_count)


def choose_mode_seeds(args: argparse.Namespace) -> dict[str, list[int]]:
    explicit = {
        "residual_zero": args.residual_zero_seed,
        "residual_random": args.residual_random_seed,
    }
    if explicit["residual_zero"] is None and explicit["residual_random"] is None:
        shared_seeds = choose_seeds(args.seed_selector, args.seed_count, args.seed_pool_max)
        return {mode: list(shared_seeds) for mode in explicit}

    mode_seeds: dict[str, list[int]] = {}
    for mode, seed in explicit.items():
        if seed is not None:
            mode_seeds[mode] = [int(seed)]
        else:
            mode_seeds[mode] = choose_seeds(args.seed_selector, args.seed_count, args.seed_pool_max)
    return mode_seeds


def build_model(args: argparse.Namespace, device: torch.device, init_mode: str) -> ResidualIOMinimalArchModel:
    cfg = MinimalArchConfig(
        image_size=args.nx,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        architecture_code=args.architecture_code,
    )
    if init_mode == "residual_zero":
        return ResidualIOMinimalArchModel(cfg, zero_init_output=True).to(device)
    if init_mode == "residual_random":
        return ResidualIOMinimalArchModel(cfg, zero_init_output=False).to(device)
    raise ValueError(f"unsupported init_mode={init_mode}")


def evaluate_fixed_batch(
    model: torch.nn.Module,
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


def compute_copy_baseline(cfg: Diffusion2DConfig, *, k: int, seeds, data_mode: str) -> float:
    inputs_np, target_np = generate_single_step_batch(cfg, k, seeds, data_mode=data_mode)
    x = torch.from_numpy(inputs_np)
    y = torch.from_numpy(target_np)
    return float(F.mse_loss(x, y).item())


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
            train_loss = row["train_loss"].strip()
            val_loss = row["val_loss"].strip()
            rows.append(
                {
                    "step": float(row["step"]),
                    "train_loss": math.nan if not train_loss else float(train_loss),
                    "val_loss": math.nan if not val_loss else float(val_loss),
                    "elapsed_s": float(row["elapsed_s"]),
                }
            )
    return rows


def train_one_run(
    args: argparse.Namespace,
    *,
    init_mode: str,
    torch_seed: int,
    device: torch.device,
    diff_cfg: Diffusion2DConfig,
    out_dir: Path,
) -> dict[str, object]:
    run_dir = ensure_dir(out_dir / init_mode / f"seed_{torch_seed}")
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model = build_model(args, device, init_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "init_mode": init_mode,
            "torch_seed": torch_seed,
            "architecture_code": canonicalize_architecture_code(args.architecture_code),
            "device_resolved": str(device),
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    )
    config_payload.update(architecture_token_counts(args.architecture_code))
    dump_json(config_path, config_payload)

    if metrics_path.exists():
        metrics_path.unlink()

    val_seeds = sample_u0_seeds_batch(args.val_base_seed, args.val_step, args.val_batch_size)
    init_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
    append_metrics_row(metrics_path, step=-1, train_loss=None, val_loss=init_val, elapsed_s=0.0)
    print(f"[compare_residual_output_init] mode={init_mode} seed={torch_seed} step=-1 val_loss={init_val:.6f}")

    best_val = init_val
    best_step = -1
    final_train = math.nan
    final_val = init_val
    t0 = time.perf_counter()
    for step in range(args.max_steps):
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
        if step % args.log_every == 0 or step == args.max_steps - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"[compare_residual_output_init] mode={init_mode} seed={torch_seed} "
                f"step={step} train_loss={final_train:.6f} elapsed_s={elapsed:.1f}"
            )

        if step % args.val_every == 0 or step == args.max_steps - 1:
            elapsed = time.perf_counter() - t0
            final_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
            append_metrics_row(metrics_path, step=step, train_loss=final_train, val_loss=final_val, elapsed_s=elapsed)
            if final_val < best_val:
                best_val = final_val
                best_step = step
            print(
                f"[compare_residual_output_init] mode={init_mode} seed={torch_seed} "
                f"step={step} val_loss={final_val:.6f} elapsed_s={elapsed:.1f}"
            )

    runtime_s = time.perf_counter() - t0
    summary = {
        "init_mode": init_mode,
        "torch_seed": torch_seed,
        "initial_val_loss": init_val,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "best_step": best_step,
        "runtime_s": runtime_s,
        "metrics_csv": str(metrics_path),
    }
    dump_json(summary_path, summary)
    return summary


def summarize_mode(run_summaries: list[dict[str, object]]) -> dict[str, float]:
    finals = np.array([float(row["final_val_loss"]) for row in run_summaries], dtype=np.float64)
    bests = np.array([float(row["best_val_loss"]) for row in run_summaries], dtype=np.float64)
    initials = np.array([float(row["initial_val_loss"]) for row in run_summaries], dtype=np.float64)
    return {
        "num_runs": float(len(run_summaries)),
        "initial_val_mean": float(initials.mean()),
        "initial_val_std": float(initials.std(ddof=0)),
        "final_val_mean": float(finals.mean()),
        "final_val_std": float(finals.std(ddof=0)),
        "final_val_min": float(finals.min()),
        "final_val_max": float(finals.max()),
        "best_val_mean": float(bests.mean()),
    }


def step_to_plot_x(step: float) -> float:
    if step < 0:
        return 0.5
    return step + 1.0


def make_plot(out_dir: Path, summaries: list[dict[str, object]], *, copy_baseline: float) -> Path:
    colors = {"residual_zero": "#d62728", "residual_random": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    train_ax, val_ax = axes

    for summary in summaries:
        rows = load_metrics(Path(str(summary["metrics_csv"])))
        seed = int(summary["torch_seed"])
        init_mode = str(summary["init_mode"])
        color = colors[init_mode]

        train_rows = [row for row in rows if row["step"] >= 0 and not math.isnan(row["train_loss"]) and row["train_loss"] > 0.0]
        val_rows = [row for row in rows if not math.isnan(row["val_loss"]) and row["val_loss"] > 0.0]

        train_x = [step_to_plot_x(row["step"]) for row in train_rows]
        train_y = [math.log10(row["train_loss"]) for row in train_rows]
        val_x = [step_to_plot_x(row["step"]) for row in val_rows]
        val_y = [math.log10(row["val_loss"]) for row in val_rows]

        label = f"{init_mode} seed={seed}"
        train_ax.plot(train_x, train_y, color=color, linewidth=1.5, alpha=0.78, label=label)
        val_ax.plot(val_x, val_y, color=color, linewidth=1.5, alpha=0.78, label=label)

    baseline_log10 = math.log10(copy_baseline)
    val_ax.axhline(baseline_log10, color="black", linestyle="--", linewidth=1.2, alpha=0.9, label="copy baseline")

    train_ax.set_title("Train Loss")
    val_ax.set_title("Validation Loss")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Optimization Step (+1; init shown at 0.5)")
        ax.set_ylabel("log10(loss)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    handles, labels = val_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.suptitle("Residual IO: Zero-Init vs Random-Init Output Projection", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    plot_path = out_dir / "residual_output_init_log_curves.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = build_argparser().parse_args()
    device = pick_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    mode_seeds = choose_mode_seeds(args)

    diff_cfg = Diffusion2DConfig(
        nx=args.nx,
        ny=args.ny,
        L=args.L,
        D=args.D,
        T=args.T,
        nt=args.nt,
        seed=args.cfg_seed,
    )
    val_seeds = sample_u0_seeds_batch(args.val_base_seed, args.val_step, args.val_batch_size)
    copy_baseline = compute_copy_baseline(diff_cfg, k=args.k, seeds=val_seeds, data_mode=args.data_mode)

    manifest = {
        "architecture_id": args.architecture_id,
        "architecture_code": canonicalize_architecture_code(args.architecture_code),
        "seeds": mode_seeds,
        "copy_baseline_val_loss": copy_baseline,
        "device_resolved": str(device),
    }
    dump_json(out_dir / "run_manifest.json", manifest)

    summaries: list[dict[str, object]] = []
    for init_mode in ("residual_zero", "residual_random"):
        for seed in mode_seeds[init_mode]:
            summary = train_one_run(
                args,
                init_mode=init_mode,
                torch_seed=seed,
                device=device,
                diff_cfg=diff_cfg,
                out_dir=out_dir,
            )
            summaries.append(summary)

    grouped = {
        "residual_zero": [row for row in summaries if row["init_mode"] == "residual_zero"],
        "residual_random": [row for row in summaries if row["init_mode"] == "residual_random"],
    }
    aggregate = {
        "architecture_id": args.architecture_id,
        "architecture_code": canonicalize_architecture_code(args.architecture_code),
        "copy_baseline_val_loss": copy_baseline,
        "device_resolved": str(device),
        "seeds": mode_seeds,
        "mode_summaries": {name: summarize_mode(rows) for name, rows in grouped.items()},
        "runs": summaries,
    }
    plot_path = make_plot(out_dir, summaries, copy_baseline=copy_baseline)
    aggregate["plot_path"] = str(plot_path)
    dump_json(out_dir / "aggregate_summary.json", aggregate)
    print(f"[compare_residual_output_init] done output_dir={out_dir}")
    print(f"[compare_residual_output_init] plot={plot_path}")


if __name__ == "__main__":
    main()
