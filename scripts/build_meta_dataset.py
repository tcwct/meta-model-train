from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate per-architecture metrics into a meta dataset table.")
    p.add_argument("--family_dir", type=str, required=True)
    p.add_argument("--output_csv", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    family_dir = Path(args.family_dir).resolve()
    manifest_path = family_dir / "family_manifest.json"
    summary_csv = family_dir / "family_summary.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing family manifest: {manifest_path}")
    if not summary_csv.exists():
        raise SystemExit(f"missing family summary: {summary_csv}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_csv = Path(args.output_csv).resolve() if args.output_csv else family_dir / "meta_dataset.csv"

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        summary_rows = {row["architecture_code"]: row for row in csv.DictReader(f)}

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

        for arch_row in manifest["architecture_rows"]:
            architecture_code = arch_row["architecture_code"]
            summary = summary_rows[architecture_code]
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
                            "k": manifest["k"],
                            "max_steps": manifest["max_steps"],
                            "val_every": manifest["val_every"],
                            "family_name": manifest["family_name"],
                        }
                    )

    print(f"[build_meta_dataset] wrote={output_csv}")


if __name__ == "__main__":
    main()
