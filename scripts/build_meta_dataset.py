from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate per-architecture metrics into a meta dataset table.")
    p.add_argument("--family_dir", type=str, nargs="+", required=True)
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument("--family_name", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    family_dirs = [Path(path).resolve() for path in args.family_dir]
    manifests: list[dict[str, object]] = []
    all_summary_rows: dict[str, dict[str, str]] = {}

    for family_dir in family_dirs:
        manifest_path = family_dir / "family_manifest.json"
        summary_csv = family_dir / "family_summary.csv"
        if not manifest_path.exists():
            raise SystemExit(f"missing family manifest: {manifest_path}")
        if not summary_csv.exists():
            raise SystemExit(f"missing family summary: {summary_csv}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)

        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_summary_rows[row["architecture_code"]] = row

    if not manifests:
        raise SystemExit("no family manifests were provided")

    base_manifest = manifests[0]
    family_name = args.family_name or base_manifest["family_name"]
    output_csv = Path(args.output_csv).resolve() if args.output_csv else family_dirs[0] / "meta_dataset.csv"

    fieldnames = (
        "architecture_id",
        "architecture_code",
        "step",
        "train_loss",
        "val_loss",
        "elapsed_s",
        "num_linear",
        "num_attention",
        "num_relu",
        "parameter_count",
        "k",
        "max_steps",
        "val_every",
        "family_name",
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        merged_arch_rows: dict[str, dict[str, str]] = {}
        for manifest in manifests:
            for arch_row in manifest["architecture_rows"]:
                merged_arch_rows[arch_row["architecture_code"]] = arch_row

        for architecture_code in sorted(
            merged_arch_rows,
            key=lambda code: int(merged_arch_rows[code]["architecture_id"]),
        ):
            arch_row = merged_arch_rows[architecture_code]
            architecture_code = arch_row["architecture_code"]
            if architecture_code not in all_summary_rows:
                raise SystemExit(f"missing summary row for architecture: {architecture_code}")
            summary = all_summary_rows[architecture_code]
            run_dir = Path(summary["run_dir"])
            metrics_path = run_dir / "metrics.csv"
            if not metrics_path.exists():
                raise SystemExit(f"missing metrics file: {metrics_path}")
            with metrics_path.open("r", newline="", encoding="utf-8") as f_metrics:
                for metric_row in csv.DictReader(f_metrics):
                    writer.writerow(
                        {
                            "architecture_id": summary["architecture_id"],
                            "architecture_code": architecture_code,
                            "step": metric_row["step"],
                            "train_loss": metric_row["train_loss"],
                            "val_loss": metric_row["val_loss"],
                            "elapsed_s": metric_row["elapsed_s"],
                            "num_linear": summary["num_linear"],
                            "num_attention": summary["num_attention"],
                            "num_relu": summary["num_relu"],
                            "parameter_count": summary["parameter_count"],
                            "k": base_manifest["k"],
                            "max_steps": base_manifest["max_steps"],
                            "val_every": base_manifest["val_every"],
                            "family_name": family_name,
                        }
                    )

    print(f"[build_meta_dataset] wrote={output_csv}")


if __name__ == "__main__":
    main()
