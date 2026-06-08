from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meta_model_train.minimal_arch_model import architecture_token_counts, enumerate_legal_architecture_codes


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enumerate legal minimal architectures and sample a subset.")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--sample_size", type=int, default=50)
    p.add_argument("--sample_seed", type=int, default=25)
    p.add_argument("--output_dir", type=str, default=str(_ROOT / "artifacts"))
    return p


def ensure_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_architecture_csv(path: Path, codes: list[str]) -> None:
    if not codes:
        raise ValueError("codes must be non-empty")
    count_keys = tuple(architecture_token_counts(codes[0]).keys())
    fieldnames = ("architecture_id", "architecture_code", *count_keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, code in enumerate(codes):
            row = {"architecture_id": idx, "architecture_code": code}
            row.update(architecture_token_counts(code))
            writer.writerow(row)


def main() -> None:
    args = build_argparser().parse_args()
    if args.depth < 1:
        raise SystemExit("depth must be >= 1")

    legal_codes = enumerate_legal_architecture_codes(expected_length=args.depth)
    if args.sample_size < 1:
        raise SystemExit("sample_size must be >= 1")
    if args.sample_size > len(legal_codes):
        raise SystemExit(f"sample_size={args.sample_size} exceeds legal architecture count={len(legal_codes)}")

    rng = random.Random(args.sample_seed)
    sampled_codes = sorted(rng.sample(legal_codes, args.sample_size))

    out_dir = ensure_dir(args.output_dir)
    all_csv = out_dir / f"all_legal_architectures_L{args.depth}.csv"
    sample_csv = out_dir / f"sampled_architectures_L{args.depth}_n{args.sample_size}_seed{args.sample_seed}.csv"
    manifest_json = out_dir / f"sample_manifest_L{args.depth}_n{args.sample_size}_seed{args.sample_seed}.json"

    write_architecture_csv(all_csv, legal_codes)
    write_architecture_csv(sample_csv, sampled_codes)

    payload = {
        "depth": args.depth,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "legal_architecture_count": len(legal_codes),
        "all_architectures_csv": str(all_csv),
        "sample_architectures_csv": str(sample_csv),
        "sampled_architecture_codes": sampled_codes,
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[enumerate_architectures] legal_count={len(legal_codes)}")
    print(f"[enumerate_architectures] wrote_all={all_csv}")
    print(f"[enumerate_architectures] wrote_sample={sample_csv}")
    print(f"[enumerate_architectures] wrote_manifest={manifest_json}")


if __name__ == "__main__":
    main()
