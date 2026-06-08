from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Train a family of sampled minimal architectures on diffusion u_k -> u_{k+1}.")
    p.add_argument("--architecture_csv", type=str, required=True)
    p.add_argument("--max_architectures", type=int, default=None)
    p.add_argument("--architecture_id_min", type=int, default=None)
    p.add_argument("--architecture_id_max", type=int, default=None)
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--D", type=float, default=0.005)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--nt", type=int, default=501)
    p.add_argument("--cfg_seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--val_base_seed", type=int, default=1000000)
    p.add_argument("--val_step", type=int, default=0)
    p.add_argument("--torch_seed", type=int, default=25)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--disable_residual", action="store_true")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--data_mode", type=str, default="slices", choices=("full", "slices"))
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--output_dir", type=str, default=os.path.join("outputs", "toy_diffusion", "meta_model_family"))
    p.add_argument("--family_name", type=str, default=None)
    p.add_argument("--skip_existing", action="store_true")
    return p


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[train_minimal_arch_family] cuda not available, using cpu")
    return torch.device("cpu")


def validate_args(args: argparse.Namespace) -> None:
    if args.nx != args.ny:
        raise SystemExit("this first family version expects nx == ny")
    if args.nx % args.patch_size != 0:
        raise SystemExit("nx and ny must be divisible by patch_size")
    if args.hidden_dim % args.num_heads != 0:
        raise SystemExit("hidden_dim must be divisible by num_heads")
    if args.batch_size < 1 or args.val_batch_size < 1:
        raise SystemExit("batch sizes must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("max_steps must be >= 1")
    if args.log_every < 1 or args.val_every < 1:
        raise SystemExit("log_every and val_every must be >= 1")
    if args.k < 0 or args.k > args.nt - 2:
        raise SystemExit(f"k must satisfy 0 <= k <= nt-2 ({args.nt - 2}), got {args.k}")


def build_family_name(args: argparse.Namespace, num_architectures: int) -> str:
    if args.family_name:
        return args.family_name
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"family_n{num_architectures}_k{args.k}_d{args.hidden_dim}_ps{args.patch_size}"
        f"_steps{args.max_steps}_seed{args.torch_seed}_{stamp}"
    )


def ensure_dir(path: str) -> str:
    path = os.path.normpath(path)
    os.makedirs(path, exist_ok=True)
    return path


def dump_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_architectures(
    csv_path: str,
    max_architectures: int | None,
    architecture_id_min: int | None,
    architecture_id_max: int | None,
) -> list[dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"architecture csv is empty: {csv_path}")
    if architecture_id_min is not None:
        rows = [row for row in rows if int(row["architecture_id"]) >= architecture_id_min]
    if architecture_id_max is not None:
        rows = [row for row in rows if int(row["architecture_id"]) <= architecture_id_max]
    if not rows:
        raise SystemExit("no architectures left after applying architecture_id filters")
    if max_architectures is not None:
        if max_architectures < 1:
            raise SystemExit("max_architectures must be >= 1 when provided")
        rows = rows[:max_architectures]
    return rows


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
    csv_path: str,
    *,
    step: int,
    architecture_code: str,
    train_loss: float | None,
    val_loss: float | None,
    elapsed_s: float,
) -> None:
    fieldnames = ("step", "architecture_code", "train_loss", "val_loss", "elapsed_s")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "step": step,
                "architecture_code": architecture_code,
                "train_loss": "" if train_loss is None else f"{train_loss:.10f}",
                "val_loss": "" if val_loss is None else f"{val_loss:.10f}",
                "elapsed_s": f"{elapsed_s:.3f}",
            }
        )


def append_summary_row(csv_path: str, row: dict[str, object]) -> None:
    fieldnames = (
        "architecture_id",
        "architecture_code",
        "num_linear",
        "num_attention",
        "num_relu",
        "parameter_count",
        "initial_val_loss",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "runtime_s",
        "run_dir",
    )
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_existing_summary_ids(csv_path: str) -> set[int]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return {int(row["architecture_id"]) for row in csv.DictReader(f)}


def train_one_architecture(
    args: argparse.Namespace,
    diff_cfg: Diffusion2DConfig,
    device: torch.device,
    family_dir: str,
    arch_row: dict[str, str],
) -> dict[str, object]:
    architecture_id = int(arch_row["architecture_id"])
    architecture_code = canonicalize_architecture_code(arch_row["architecture_code"])
    run_name = f"arch_{architecture_id:03d}_{architecture_code.replace('-', '')}"
    out_dir = ensure_dir(os.path.join(family_dir, run_name))
    csv_path = os.path.join(out_dir, "metrics.csv")

    if args.skip_existing and os.path.exists(csv_path):
        summary_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)

    model_cfg = MinimalArchConfig(
        image_size=args.nx,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        architecture_code=architecture_code,
        use_residual=not args.disable_residual,
    )
    model = MinimalArchModel(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg_payload = vars(args).copy()
    cfg_payload["architecture_id"] = architecture_id
    cfg_payload["architecture_code"] = model.architecture_code
    cfg_payload["architecture_tokens"] = model.architecture_tokens
    cfg_payload["use_residual"] = model.use_residual
    cfg_payload["device_resolved"] = str(device)
    cfg_payload["parameter_count"] = parameter_count
    cfg_payload.update(architecture_token_counts(model.architecture_code))
    dump_json(os.path.join(out_dir, "config.json"), cfg_payload)

    val_seeds = sample_u0_seeds_batch(args.val_base_seed, args.val_step, args.val_batch_size)
    init_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
    append_metrics_row(
        csv_path,
        step=-1,
        architecture_code=model.architecture_code,
        train_loss=None,
        val_loss=init_val,
        elapsed_s=0.0,
    )

    t0 = time.perf_counter()
    best_val = init_val
    final_val = init_val
    last_train_loss: float | None = None

    for step in range(args.max_steps):
        model.train()
        seeds = sample_u0_seeds_batch(args.base_seed, step, args.batch_size)
        inputs_np, target_np = generate_single_step_batch(diff_cfg, args.k, seeds, data_mode=args.data_mode)
        x = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()

        last_train_loss = float(loss.item())
        if step % args.log_every == 0 or step == args.max_steps - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"[train_minimal_arch_family] arch={architecture_id:03d} code={model.architecture_code} "
                f"step={step} train_loss={last_train_loss:.6f} elapsed_s={elapsed:.1f}"
            )

        if step % args.val_every == 0 or step == args.max_steps - 1:
            elapsed = time.perf_counter() - t0
            final_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
            best_val = min(best_val, final_val)
            append_metrics_row(
                csv_path,
                step=step,
                architecture_code=model.architecture_code,
                train_loss=last_train_loss,
                val_loss=final_val,
                elapsed_s=elapsed,
            )
            print(
                f"[train_minimal_arch_family] arch={architecture_id:03d} code={model.architecture_code} "
                f"step={step} val_loss={final_val:.6f} elapsed_s={elapsed:.1f}"
            )

    runtime_s = time.perf_counter() - t0
    summary = {
        "architecture_id": architecture_id,
        "architecture_code": model.architecture_code,
        "parameter_count": parameter_count,
        "initial_val_loss": init_val,
        "final_train_loss": last_train_loss,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "runtime_s": runtime_s,
        "run_dir": out_dir,
    }
    summary.update(architecture_token_counts(model.architecture_code))
    dump_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)

    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)

    arch_rows = load_architectures(
        args.architecture_csv,
        args.max_architectures,
        args.architecture_id_min,
        args.architecture_id_max,
    )
    device = pick_device(args.device)
    family_name = build_family_name(args, num_architectures=len(arch_rows))
    family_dir = ensure_dir(os.path.join(args.output_dir, family_name))
    summary_csv = os.path.join(family_dir, "family_summary.csv")
    existing_summary_ids = read_existing_summary_ids(summary_csv)

    diff_cfg = Diffusion2DConfig(
        nx=args.nx,
        ny=args.ny,
        L=args.L,
        D=args.D,
        T=args.T,
        nt=args.nt,
        seed=args.cfg_seed,
    )
    family_manifest = vars(args).copy()
    family_manifest["device_resolved"] = str(device)
    family_manifest["family_name"] = family_name
    family_manifest["num_architectures"] = len(arch_rows)
    family_manifest["architecture_rows"] = arch_rows
    dump_json(os.path.join(family_dir, "family_manifest.json"), family_manifest)

    t0 = time.perf_counter()
    for idx, arch_row in enumerate(arch_rows, start=1):
        print(
            f"[train_minimal_arch_family] start {idx}/{len(arch_rows)} "
            f"arch_id={arch_row['architecture_id']} code={arch_row['architecture_code']}"
        )
        summary = train_one_architecture(args, diff_cfg, device, family_dir, arch_row)
        arch_id = int(summary["architecture_id"])
        if arch_id in existing_summary_ids:
            continue
        append_summary_row(summary_csv, summary)
        existing_summary_ids.add(arch_id)

    total_runtime = time.perf_counter() - t0
    print(f"[train_minimal_arch_family] done family_dir={family_dir} total_runtime_s={total_runtime:.1f}")


if __name__ == "__main__":
    main()
