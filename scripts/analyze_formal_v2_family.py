from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze formal_v2 architecture family results.")
    parser.add_argument("--family_root", type=str, required=True, help="Root directory that contains shardXX/family_summary.csv files.")
    parser.add_argument("--meta_csv", type=str, required=True, help="Merged meta_dataset.csv path.")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_family_summary_paths(family_root: Path) -> list[Path]:
    paths = sorted(family_root.glob("shard*/family_summary.csv"))
    if not paths:
        raise SystemExit(f"no family_summary.csv files found under {family_root}")
    return paths


def compact_code(code: str) -> str:
    return code.replace("-", "")


def count_switches(code: str) -> int:
    return sum(1 for left, right in zip(code, code[1:]) if left != right)


def longest_run(code: str, token: str) -> int:
    best = 0
    current = 0
    for ch in code:
        if ch == token:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def attention_center(code: str) -> float | None:
    positions = [idx + 1 for idx, ch in enumerate(code) if ch == "A"]
    if not positions:
        return None
    return statistics.mean(positions)


def half_balance_group(code: str) -> str:
    front = sum(1 for ch in code[:4] if ch == "A")
    back = sum(1 for ch in code[4:] if ch == "A")
    if front > back:
        return "front-heavy"
    if back > front:
        return "back-heavy"
    return "balanced"


def transition_group(code: str) -> str:
    switches = count_switches(code)
    if switches <= 2:
        return "clustered"
    if switches <= 4:
        return "mixed"
    return "alternating"


def enrich_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        code = compact_code(row["architecture_code"])
        enriched.append(
            {
                **row,
                "architecture_code_compact": code,
                "num_attention": int(row["num_attention"]),
                "num_mlp": int(row["num_mlp"]),
                "parameter_count": int(row["parameter_count"]),
                "best_val_loss": float(row["best_val_loss"]),
                "final_val_loss": float(row["final_val_loss"]),
                "final_train_loss": float(row["final_train_loss"]),
                "initial_val_loss": float(row["initial_val_loss"]),
                "runtime_s": float(row["runtime_s"]),
                "attention_center": attention_center(code),
                "switch_count": count_switches(code),
                "max_attention_run": longest_run(code, "A"),
                "max_mlp_run": longest_run(code, "M"),
                "half_balance_group": half_balance_group(code),
                "transition_group": transition_group(code),
            }
        )
    return enriched


def load_summary_rows(family_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, str]] = []
    for path in iter_family_summary_paths(family_root):
        rows.extend(read_csv_rows(path))
    if not rows:
        raise SystemExit("family summaries are empty")
    return enrich_summary_rows(rows)


def jittered_x(values: list[int]) -> list[float]:
    offsets = [-0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18]
    counters: dict[int, int] = defaultdict(int)
    xs: list[float] = []
    for value in values:
        idx = counters[value] % len(offsets)
        counters[value] += 1
        xs.append(value + offsets[idx])
    return xs


def summarize_by_key(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(float(row["best_val_loss"]))

    summary: list[dict[str, object]] = []
    for group_key in sorted(groups, key=lambda item: (isinstance(item, str), item)):
        losses = groups[group_key]
        summary.append(
            {
                key: group_key,
                "count": len(losses),
                "mean_best_val_loss": statistics.mean(losses),
                "median_best_val_loss": statistics.median(losses),
                "min_best_val_loss": min(losses),
                "max_best_val_loss": max(losses),
            }
        )
    return summary


def plot_best_loss_vs_num_attention(rows: list[dict[str, object]], output_png: Path) -> None:
    xs = [int(row["num_attention"]) for row in rows]
    ys = [float(row["best_val_loss"]) for row in rows]
    jitter = jittered_x(xs)
    groups = summarize_by_key(rows, "num_attention")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for ax, use_log in zip(axes, [False, True]):
        ax.scatter(jitter, ys, alpha=0.75, s=34, label="architectures")
        ax.plot(
            [int(group["num_attention"]) for group in groups],
            [float(group["mean_best_val_loss"]) for group in groups],
            linewidth=2.2,
            marker="o",
            label="group mean",
        )
        ax.plot(
            [int(group["num_attention"]) for group in groups],
            [float(group["median_best_val_loss"]) for group in groups],
            linewidth=2.0,
            marker="s",
            linestyle="--",
            label="group median",
        )
        ax.set_xlabel("number of attention blocks")
        ax.set_ylabel("best validation loss")
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(set(xs)))
        if use_log:
            ax.set_yscale("log")
            ax.set_title("Best val loss vs attention count (log y)")
        else:
            ax.set_title("Best val loss vs attention count")

    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_position_effects(rows: list[dict[str, object]], output_png: Path) -> None:
    centers = [row["attention_center"] for row in rows if row["attention_center"] is not None]
    losses = [float(row["best_val_loss"]) for row in rows if row["attention_center"] is not None]
    counts = [int(row["num_attention"]) for row in rows if row["attention_center"] is not None]
    switches = [int(row["switch_count"]) for row in rows if row["attention_center"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    scatter0 = axes[0].scatter(centers, losses, c=counts, cmap="viridis", s=48, alpha=0.82)
    axes[0].set_title("Best val loss vs attention center")
    axes[0].set_xlabel("mean attention layer index")
    axes[0].set_ylabel("best validation loss")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    fig.colorbar(scatter0, ax=axes[0], label="num attention")

    scatter1 = axes[1].scatter(switches, losses, c=counts, cmap="viridis", s=48, alpha=0.82)
    axes[1].set_title("Best val loss vs A/M switch count")
    axes[1].set_xlabel("number of A/M switches")
    axes[1].set_ylabel("best validation loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    fig.colorbar(scatter1, ax=axes[1], label="num attention")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_meta_rows(meta_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv_rows(meta_csv):
        rows.append(
            {
                "architecture_code": compact_code(row["architecture_code"]),
                "step": int(row["step"]),
                "train_loss": float(row["train_loss"]) if row["train_loss"] else math.nan,
                "val_loss": float(row["val_loss"]) if row["val_loss"] else math.nan,
            }
        )
    return rows


def plot_pattern_group_curves(
    meta_rows: list[dict[str, object]],
    summary_by_code: dict[str, dict[str, object]],
    group_key: str,
    group_order: list[str],
    title: str,
    output_png: Path,
) -> None:
    group_curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen_architectures: dict[str, set[str]] = defaultdict(set)

    for row in meta_rows:
        code = str(row["architecture_code"])
        if code not in summary_by_code:
            continue
        group = str(summary_by_code[code][group_key])
        step = int(row["step"])
        if step < 0:
            continue
        group_curves[group][step].append(float(row["val_loss"]))
        seen_architectures[group].add(code)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    for group in group_order:
        if group not in group_curves:
            continue
        steps = sorted(group_curves[group])
        mean_curve = [statistics.mean(group_curves[group][step]) for step in steps]
        ax.plot(
            steps,
            mean_curve,
            linewidth=2.2,
            label=f"{group} (n={len(seen_architectures[group])})",
        )

    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise_marginal_effect(rows: list[dict[str, object]], output_png: Path) -> None:
    attn_groups: dict[int, list[float]] = defaultdict(list)
    mlp_groups: dict[int, list[float]] = defaultdict(list)

    for row in rows:
        code = str(row["architecture_code_compact"])
        loss = float(row["best_val_loss"])
        for idx, ch in enumerate(code, start=1):
            if ch == "A":
                attn_groups[idx].append(loss)
            else:
                mlp_groups[idx].append(loss)

    layers = sorted(attn_groups)
    attn_means = [statistics.mean(attn_groups[layer]) for layer in layers]
    mlp_means = [statistics.mean(mlp_groups[layer]) for layer in layers]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(layers, attn_means, marker="o", linewidth=2.2, label="layer is attention")
    ax.plot(layers, mlp_means, marker="s", linewidth=2.2, linestyle="--", label="layer is mlp")
    ax.set_title("Marginal best-loss effect of each layer choice")
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean best validation loss")
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_artifacts(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    top_rows = sorted(rows, key=lambda row: float(row["best_val_loss"]))[:10]
    top_payload = [
        {
            "architecture_code": row["architecture_code"],
            "best_val_loss": row["best_val_loss"],
            "num_attention": row["num_attention"],
            "attention_center": row["attention_center"],
            "switch_count": row["switch_count"],
            "half_balance_group": row["half_balance_group"],
            "transition_group": row["transition_group"],
        }
        for row in top_rows
    ]

    by_attention = summarize_by_key(rows, "num_attention")
    by_half_balance = summarize_by_key(rows, "half_balance_group")
    by_transition = summarize_by_key(rows, "transition_group")

    summary_json = {
        "num_architectures": len(rows),
        "top10": top_payload,
        "by_num_attention": by_attention,
        "by_half_balance_group": by_half_balance,
        "by_transition_group": by_transition,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    with (output_dir / "group_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_type",
                "group_name",
                "count",
                "mean_best_val_loss",
                "median_best_val_loss",
                "min_best_val_loss",
                "max_best_val_loss",
            ],
        )
        writer.writeheader()
        for group_type, rows_list in [
            ("num_attention", by_attention),
            ("half_balance_group", by_half_balance),
            ("transition_group", by_transition),
        ]:
            for row in rows_list:
                group_name = row.get(group_type, "")
                writer.writerow(
                    {
                        "group_type": group_type,
                        "group_name": group_name,
                        "count": row["count"],
                        "mean_best_val_loss": row["mean_best_val_loss"],
                        "median_best_val_loss": row["median_best_val_loss"],
                        "min_best_val_loss": row["min_best_val_loss"],
                        "max_best_val_loss": row["max_best_val_loss"],
                    }
                )


def main() -> None:
    args = build_argparser().parse_args()
    family_root = Path(args.family_root).resolve()
    meta_csv = Path(args.meta_csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    summary_rows = load_summary_rows(family_root)
    summary_by_code = {str(row["architecture_code_compact"]): row for row in summary_rows}
    meta_rows = load_meta_rows(meta_csv)

    plot_best_loss_vs_num_attention(summary_rows, output_dir / "best_val_vs_num_attention.png")
    plot_position_effects(summary_rows, output_dir / "best_val_vs_position_effects.png")
    plot_pattern_group_curves(
        meta_rows,
        summary_by_code,
        group_key="half_balance_group",
        group_order=["front-heavy", "balanced", "back-heavy"],
        title="Mean validation curves grouped by attention placement",
        output_png=output_dir / "val_curves_by_half_balance_group.png",
    )
    plot_pattern_group_curves(
        meta_rows,
        summary_by_code,
        group_key="transition_group",
        group_order=["clustered", "mixed", "alternating"],
        title="Mean validation curves grouped by A/M alternation",
        output_png=output_dir / "val_curves_by_transition_group.png",
    )
    plot_layerwise_marginal_effect(summary_rows, output_dir / "layerwise_marginal_effect.png")
    write_summary_artifacts(summary_rows, output_dir)

    print(f"[analyze_formal_v2_family] wrote_dir={output_dir}")


if __name__ == "__main__":
    main()
