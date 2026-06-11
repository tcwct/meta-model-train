from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_model_train.diffusion_dataset import Diffusion2DConfig, generate_single_step_batch, sample_u0_seeds_batch


ARCH_TOKENS = ("L", "A", "R")


def canonicalize_architecture_code(code: str) -> str:
    compact = code.replace("-", "").replace(",", "").replace(" ", "").upper()
    if not compact:
        raise ValueError("architecture_code must be non-empty")
    return "-".join(compact)


def parse_architecture_code(code: str) -> list[str]:
    compact = code.replace("-", "").replace(",", "").replace(" ", "").upper()
    tokens = list(compact)
    if not tokens:
        raise ValueError("architecture_code must be non-empty")
    bad = [tok for tok in tokens if tok not in ARCH_TOKENS]
    if bad:
        raise ValueError(f"architecture_code contains invalid tokens: {bad}; allowed tokens are {ARCH_TOKENS}")
    return tokens


def validate_architecture_tokens(tokens: Iterable[str], expected_length: int) -> list[str]:
    toks = list(tokens)
    if len(toks) != expected_length:
        raise ValueError(f"architecture must have exactly {expected_length} tokens, got {len(toks)}")
    if toks[0] == "R":
        raise ValueError("first architecture token may not be ReLU")
    for left, right in zip(toks, toks[1:]):
        if left == "R" and right == "R":
            raise ValueError("adjacent ReLU tokens are not allowed")
    return toks


def architecture_token_counts(code: str) -> dict[str, int]:
    tokens = parse_architecture_code(code)
    return {
        "num_linear": sum(tok == "L" for tok in tokens),
        "num_attention": sum(tok == "A" for tok in tokens),
        "num_relu": sum(tok == "R" for tok in tokens),
    }


def _periodic_delta_indices(length: int, device: torch.device) -> torch.Tensor:
    qpos = torch.arange(length, device=device)[:, None]
    kpos = torch.arange(length, device=device)[None, :]
    return (kpos - qpos) % length


class PeriodicRelativeBias1D(nn.Module):
    def __init__(self, length: int, n_heads: int):
        super().__init__()
        self.length = int(length)
        self.n_heads = int(n_heads)
        self.table = nn.Parameter(torch.zeros(self.length, self.n_heads))

    def forward(self) -> torch.Tensor:
        idx = _periodic_delta_indices(self.length, self.table.device)
        bias = self.table[idx.long()]
        return bias.permute(2, 0, 1).unsqueeze(0).contiguous()


class TokenLinearOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ReLUOp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class AxialAttentionOp(nn.Module):
    def __init__(self, dim: int, height: int, width: int, num_heads: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_row = nn.Linear(dim, 3 * dim, bias=True)
        self.out_row = nn.Linear(dim, dim, bias=True)
        self.qkv_col = nn.Linear(dim, 3 * dim, bias=True)
        self.out_col = nn.Linear(dim, dim, bias=True)

        self.row_bias = PeriodicRelativeBias1D(width, num_heads)
        self.col_bias = PeriodicRelativeBias1D(height, num_heads)

    def _apply_attention(self, x: torch.Tensor, qkv_layer: nn.Linear, out_layer: nn.Linear, bias: torch.Tensor) -> torch.Tensor:
        batch_like, seq_len, dim = x.shape
        qkv = qkv_layer(x).reshape(batch_like, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        out = out.permute(0, 2, 1, 3).reshape(batch_like, seq_len, dim)
        return out_layer(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, dim = x.shape
        if height != self.height or width != self.width or dim != self.dim:
            raise ValueError(f"expected input (B,{self.height},{self.width},{self.dim}), got {tuple(x.shape)}")

        row_in = x.reshape(batch * height, width, dim)
        row_out = self._apply_attention(row_in, self.qkv_row, self.out_row, self.row_bias())
        row_out = row_out.reshape(batch, height, width, dim)

        col_in = x.permute(0, 2, 1, 3).reshape(batch * width, height, dim)
        col_out = self._apply_attention(col_in, self.qkv_col, self.out_col, self.col_bias())
        col_out = col_out.reshape(batch, width, height, dim).permute(0, 2, 1, 3).contiguous()

        return 0.5 * (row_out + col_out)


def patchify_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if channels != 1:
        raise ValueError(f"expected 1 input channel, got {channels}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"height={height} and width={width} must be divisible by patch_size={patch_size}")
    hp = height // patch_size
    wp = width // patch_size
    x = x.reshape(batch, channels, hp, patch_size, wp, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(batch, hp, wp, channels * patch_size * patch_size)


def unpatchify_2d(tokens: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, hp, wp, patch_dim = tokens.shape
    expected_patch_dim = patch_size * patch_size
    if patch_dim != expected_patch_dim:
        raise ValueError(f"expected patch_dim={expected_patch_dim}, got {patch_dim}")
    x = tokens.reshape(batch, hp, wp, 1, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, 1, hp * patch_size, wp * patch_size)


@dataclass(frozen=True)
class MinimalArchConfigV3:
    image_size: int = 16
    patch_size: int = 2
    hidden_dim: int = 8
    num_heads: int = 1
    architecture_code: str = "L-R-L-A-L-R"
    use_residual: bool = True


class MinimalArchModelV3(nn.Module):
    def __init__(self, cfg: MinimalArchConfigV3):
        super().__init__()
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.cfg = cfg
        self.patch_dim = cfg.patch_size * cfg.patch_size
        self.hp = cfg.image_size // cfg.patch_size
        self.wp = cfg.image_size // cfg.patch_size

        tokens = validate_architecture_tokens(parse_architecture_code(cfg.architecture_code), expected_length=6)
        self.architecture_tokens = tokens
        self.architecture_code = canonicalize_architecture_code(cfg.architecture_code)
        self.use_residual = bool(cfg.use_residual)

        self.input_proj = nn.Linear(self.patch_dim, cfg.hidden_dim, bias=True)
        self.output_proj = nn.Linear(cfg.hidden_dim, self.patch_dim, bias=True)

        ops: list[nn.Module] = []
        for tok in tokens:
            if tok == "L":
                ops.append(TokenLinearOp(cfg.hidden_dim))
            elif tok == "A":
                ops.append(AxialAttentionOp(cfg.hidden_dim, self.hp, self.wp, num_heads=cfg.num_heads))
            elif tok == "R":
                ops.append(ReLUOp())
            else:
                raise AssertionError(f"unreachable token: {tok}")
        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected input rank 4, got shape {tuple(x.shape)}")
        patches = patchify_2d(x, self.cfg.patch_size)
        hidden = self.input_proj(patches)
        for op in self.ops:
            update = op(hidden)
            if self.use_residual:
                hidden = hidden + update
            else:
                hidden = update
        pred_patches = self.output_proj(hidden)
        return unpatchify_2d(pred_patches, self.cfg.patch_size)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local seed stability sweeps for architecture L-A-L-L-A-R.")
    parser.add_argument("--architecture_id", type=int, default=196)
    parser.add_argument("--architecture_code", type=str, default="L-A-L-L-A-R")
    parser.add_argument("--seed_selector", type=int, default=20260609)
    parser.add_argument("--short_seed_count", type=int, default=4)
    parser.add_argument("--seed_pool_max", type=int, default=999)
    parser.add_argument("--long_steps", type=int, default=4000)
    parser.add_argument("--short_steps", type=int, default=1000)
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
        default=os.path.join("outputs", "toy_diffusion", "single_arch_seed_stability", "lallar_v3_local"),
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
    print("[run_lallar_seed_stability] cuda not available, using cpu")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_fixed_batch(
    model: MinimalArchModelV3,
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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

    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model_cfg = MinimalArchConfigV3(
        image_size=args.nx,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        architecture_code=args.architecture_code,
        use_residual=True,
    )
    model = MinimalArchModelV3(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    val_seeds = sample_u0_seeds_batch(args.val_base_seed, args.val_step, args.val_batch_size)
    init_val = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
    append_metrics_row(metrics_path, step=-1, train_loss=None, val_loss=init_val, elapsed_s=0.0)
    print(f"[run_lallar_seed_stability] {label} step=-1 val_loss={init_val:.6f}")

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

        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()

        final_train = float(loss.item())
        if step % args.log_every == 0 or step == max_steps - 1:
            elapsed = time.perf_counter() - t0
            print(f"[run_lallar_seed_stability] {label} step={step} train_loss={final_train:.6f} elapsed_s={elapsed:.1f}")

        if step % args.val_every == 0 or step == max_steps - 1:
            elapsed = time.perf_counter() - t0
            val_loss = evaluate_fixed_batch(model, diff_cfg, args.k, val_seeds, device, data_mode=args.data_mode)
            final_val = val_loss
            append_metrics_row(metrics_path, step=step, train_loss=final_train, val_loss=val_loss, elapsed_s=elapsed)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
            print(f"[run_lallar_seed_stability] {label} step={step} val_loss={val_loss:.6f} elapsed_s={elapsed:.1f}")

    runtime_s = time.perf_counter() - t0
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
    }
    dump_json(summary_path, summary)
    return summary


def choose_seeds(selector: int, short_seed_count: int, seed_pool_max: int) -> tuple[list[int], int]:
    if short_seed_count < 1:
        raise ValueError("short_seed_count must be >= 1")
    rng = random.Random(selector)
    seeds = rng.sample(range(1, seed_pool_max + 1), short_seed_count + 1)
    return seeds[:short_seed_count], seeds[-1]


def load_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            val_loss = row["val_loss"].strip()
            if step < 0 or not val_loss:
                continue
            rows.append(
                {
                    "step": float(step),
                    "val_loss": float(val_loss),
                }
            )
    return rows


def make_plot(out_dir: Path, run_specs: list[dict[str, object]]) -> Path:
    plt.figure(figsize=(9.5, 6.2))
    for spec in run_specs:
        rows = load_metrics(Path(str(spec["metrics_csv"])))
        x = [row["step"] + 1.0 for row in rows]
        y = [row["val_loss"] for row in rows]
        linewidth = 2.4 if bool(spec["is_long"]) else 1.8
        alpha = 0.95 if bool(spec["is_long"]) else 0.8
        label = f"seed={spec['torch_seed']}, steps={spec['max_steps']}"
        plt.plot(x, y, label=label, linewidth=linewidth, alpha=alpha)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Optimization Step + 1 (log scale)")
    plt.ylabel("Validation Loss (log scale)")
    plt.title("L-A-L-L-A-R Seed Stability and Long-Run Behavior")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "lallar_seed_stability_loglog.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return plot_path


def main() -> None:
    args = build_argparser().parse_args()
    device = pick_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    short_seeds, long_seed = choose_seeds(args.seed_selector, args.short_seed_count, args.seed_pool_max)
    seed_manifest = {
        "seed_selector": args.seed_selector,
        "short_seeds": short_seeds,
        "long_seed": long_seed,
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

    run_specs: list[dict[str, object]] = []
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
        summary["is_long"] = False
        run_specs.append(summary)

    long_label = f"long_seed_{long_seed}"
    long_summary = train_one_seed(
        args,
        torch_seed=long_seed,
        max_steps=args.long_steps,
        label=long_label,
        device=device,
        diff_cfg=diff_cfg,
        out_dir=out_dir,
    )
    long_summary["is_long"] = True
    run_specs.append(long_summary)

    plot_path = make_plot(out_dir, run_specs)
    aggregate = {
        "architecture_id": args.architecture_id,
        "architecture_code": canonicalize_architecture_code(args.architecture_code),
        "device_resolved": str(device),
        "runs": run_specs,
        "plot_path": str(plot_path),
    }
    dump_json(out_dir / "aggregate_summary.json", aggregate)
    print(f"[run_lallar_seed_stability] done output_dir={out_dir}")
    print(f"[run_lallar_seed_stability] plot={plot_path}")


if __name__ == "__main__":
    main()
