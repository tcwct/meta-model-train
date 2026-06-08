from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a minimal end-to-end smoke pipeline for the meta-model project.")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "auto"))
    p.add_argument("--sample_size", type=int, default=4)
    p.add_argument("--max_architectures", type=int, default=2)
    p.add_argument("--smoke_steps", type=int, default=2)
    p.add_argument("--family_steps", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--pipeline_name", type=str, default=None)
    p.add_argument("--output_root", type=str, default=str(_ROOT / "outputs" / "smoke_pipeline"))
    return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(args: list[str]) -> None:
    print(f"[run_smoke_pipeline] cmd={' '.join(args)}")
    subprocess.run(args, check=True, cwd=_ROOT)


def main() -> None:
    args = build_argparser().parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pipeline_name = args.pipeline_name or f"pipeline_{stamp}"

    pipeline_root = ensure_dir(Path(args.output_root).resolve() / pipeline_name)
    artifact_dir = ensure_dir(pipeline_root / "artifacts")
    smoke_output_dir = ensure_dir(pipeline_root / "smoke")
    family_output_root = ensure_dir(pipeline_root / "family_runs")

    smoke_run_name = "smoke"
    family_name = "family"
    architecture_csv = artifact_dir / f"sampled_architectures_L8_n{args.sample_size}_seed25.csv"
    family_dir = family_output_root / family_name

    python_exe = sys.executable
    scripts_dir = _ROOT / "scripts"

    run_command(
        [
            python_exe,
            str(scripts_dir / "train_minimal_arch_smoke.py"),
            "--device",
            args.device,
            "--max_steps",
            str(args.smoke_steps),
            "--batch_size",
            str(args.batch_size),
            "--val_batch_size",
            str(args.val_batch_size),
            "--log_every",
            str(args.log_every),
            "--val_every",
            str(args.val_every),
            "--output_dir",
            str(smoke_output_dir),
            "--run_name",
            smoke_run_name,
        ]
    )

    run_command(
        [
            python_exe,
            str(scripts_dir / "enumerate_architectures.py"),
            "--sample_size",
            str(args.sample_size),
            "--output_dir",
            str(artifact_dir),
        ]
    )

    run_command(
        [
            python_exe,
            str(scripts_dir / "train_minimal_arch_family.py"),
            "--architecture_csv",
            str(architecture_csv),
            "--max_architectures",
            str(args.max_architectures),
            "--device",
            args.device,
            "--max_steps",
            str(args.family_steps),
            "--batch_size",
            str(args.batch_size),
            "--val_batch_size",
            str(args.val_batch_size),
            "--log_every",
            str(args.log_every),
            "--val_every",
            str(args.val_every),
            "--output_dir",
            str(family_output_root),
            "--family_name",
            family_name,
        ]
    )

    run_command(
        [
            python_exe,
            str(scripts_dir / "build_meta_dataset.py"),
            "--family_dir",
            str(family_dir),
        ]
    )

    manifest = {
        "pipeline_name": pipeline_name,
        "pipeline_root": str(pipeline_root),
        "smoke_run_dir": str(smoke_output_dir / smoke_run_name),
        "artifact_dir": str(artifact_dir),
        "architecture_csv": str(architecture_csv),
        "family_dir": str(family_dir),
        "meta_dataset_csv": str(family_dir / "meta_dataset.csv"),
        "device": args.device,
        "sample_size": args.sample_size,
        "max_architectures": args.max_architectures,
        "smoke_steps": args.smoke_steps,
        "family_steps": args.family_steps,
    }
    manifest_path = pipeline_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[run_smoke_pipeline] wrote_manifest={manifest_path}")
    print(f"[run_smoke_pipeline] pipeline_root={pipeline_root}")


if __name__ == "__main__":
    main()
