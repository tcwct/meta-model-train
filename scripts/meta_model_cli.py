from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


COMMAND_TO_SCRIPT = {
    "check-env": "check_environment.py",
    "smoke-train": "train_minimal_arch_smoke.py",
    "enumerate": "enumerate_architectures.py",
    "family-train": "train_minimal_arch_family.py",
    "single-arch-seed-scaling": "run_single_arch_seed_scaling.py",
    "build-meta-dataset": "build_meta_dataset.py",
    "plot-family-curves": "plot_family_curves.py",
    "plot-meta-spread": "plot_meta_dataset_spread.py",
    "run-baselines": "run_meta_baselines.py",
    "smoke-pipeline": "run_smoke_pipeline.py",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified command entrypoint for the meta-model training project."
    )
    parser.add_argument("command", choices=sorted(COMMAND_TO_SCRIPT))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the underlying script.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    script_name = COMMAND_TO_SCRIPT[args.command]
    script_path = _SCRIPTS / script_name
    cmd = [sys.executable, str(script_path), *args.args]
    print(f"[meta_model_cli] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_ROOT)


if __name__ == "__main__":
    main()
