#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def find_latest_summary(results_dir):
    candidates = sorted(results_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def extract_avg_input_length(summary):
    length_stats = summary.get("length_statistics")
    if not isinstance(length_stats, dict):
        return None
    subjects = length_stats.get("subjects")
    if isinstance(subjects, dict) and subjects:
        vals = [safe_float(v.get("avg_input_length")) for v in subjects.values()]
        return mean(vals)
    subcats = length_stats.get("subcategories")
    if isinstance(subcats, dict) and subcats:
        vals = [safe_float(v.get("avg_input_length")) for v in subcats.values()]
        return mean(vals)
    return None


def quant_bits_from_config(kv_quant_config):
    if not kv_quant_config or not kv_quant_config.get("enabled", False):
        return 16, "fp16"
    scheme = kv_quant_config.get("scheme", "int8")
    if scheme == "int4":
        return 4, "int4"
    if scheme == "int8":
        return 8, "int8"
    return 16, scheme


def load_model_stats(model_name, override):
    if override:
        return override
    try:
        from transformers import AutoConfig
    except Exception as exc:
        die(f"transformers not available to load model config: {exc}")
    cfg = AutoConfig.from_pretrained(model_name)
    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
    if not all([hidden_size, num_layers, num_heads, num_kv_heads]):
        die(f"Missing model stats for {model_name}")
    head_dim = hidden_size // num_heads
    return {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }


def bytes_per_token(model_stats, bits_per_elem):
    bytes_per_elem = bits_per_elem / 8.0
    return 2.0 * model_stats["num_layers"] * model_stats["num_kv_heads"] * model_stats["head_dim"] * bytes_per_elem


def iter_run_roots(root):
    for sub in ("step_0_baselines", "step_1_kv_ptq"):
        base = root / sub
        if not base.exists():
            continue
        for run_root in sorted(base.iterdir()):
            if run_root.is_dir():
                yield run_root


def parse_run(run_root):
    manifest = {}
    step1_manifest = run_root / "manifests" / "step_1_manifest.json"
    if step1_manifest.exists():
        manifest = json.loads(step1_manifest.read_text())
    run_manifest = run_root / "manifests" / "run_manifest.json"
    if run_manifest.exists():
        try:
            manifest.update(json.loads(run_manifest.read_text()))
        except Exception:
            pass

    results_root = run_root / "results"
    if not results_root.exists():
        return []

    rows = []
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        summary_path = find_latest_summary(dataset_dir)
        if summary_path is None:
            continue
        summary = json.loads(summary_path.read_text())
        cfg_path = run_root / "configs" / f"{dataset_dir.name}.yaml"
        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        model_cfg = cfg.get("model", {}).get("rosetta_config", {})
        base_model = model_cfg.get("base_model", "Qwen/Qwen3-0.6B")
        kv_quant_config = model_cfg.get("kv_quant_config") or manifest.get("kv_quant_config")
        bits, scheme = quant_bits_from_config(kv_quant_config)
        eval_cfg = cfg.get("eval", {})
        proportion = eval_cfg.get("kv_cache_proportion", manifest.get("kv_cache_proportion", 1.0))
        order_mode = eval_cfg.get("kv_cache_order_mode", manifest.get("kv_cache_order_mode", "front"))
        rows.append(
            {
                "run_root": str(run_root),
                "run_tag": run_root.name,
                "dataset": dataset_dir.name,
                "base_model": base_model,
                "kv_quant_scheme": scheme,
                "kv_cache_proportion": float(proportion),
                "kv_cache_order_mode": order_mode,
                "accuracy": safe_float(summary.get("overall_accuracy")),
                "avg_input_length": extract_avg_input_length(summary),
            }
        )
    return rows


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "dataset",
        "kv_quant_scheme",
        "kv_cache_proportion",
        "kv_cache_order_mode",
        "base_model",
        "avg_input_length",
        "bytes_per_sequence",
        "bytes_per_sequence_mib",
        "accuracy",
        "run_root",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_budget_curve(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot (matplotlib missing): {exc}")
        return

    datasets = sorted({row["dataset"] for row in rows})
    scheme_colors = {"fp16": "#333333", "int8": "#1f77b4", "int4": "#ff7f0e"}
    order_markers = {"front": "o", "back": "s"}

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in sorted(rows, key=lambda r: r["bytes_per_sequence"]):
            if row["dataset"] != dataset:
                continue
            scheme = row["kv_quant_scheme"]
            order = row["kv_cache_order_mode"]
            ax.scatter(
                row["bytes_per_sequence"],
                row["accuracy"],
                c=scheme_colors.get(scheme, "#999999"),
                marker=order_markers.get(order, "x"),
                label=f"{scheme}/{order}",
            )
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=9)
        ax.set_xlabel("Bytes per sequence (approx)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs Bytes ({dataset})")
        ax.grid(True, linestyle="--", alpha=0.4)
        out_path = out_dir / f"budget_curve_{dataset}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy vs bytes curves for C2C runs.")
    parser.add_argument(
        "--runs-root",
        action="append",
        default=[],
        help="Root directory containing step_0_baselines/ and step_1_kv_ptq/ (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default="quantization/analysis/m4_budget_curve",
        help="Directory to write CSV/plots",
    )
    parser.add_argument(
        "--assume-input-length",
        type=float,
        default=None,
        help="Fallback avg input length if summaries lack length stats",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating PNG plots")
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic CSV/plot for validation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        rows = [
            {
                "run_tag": "demo_fp16",
                "dataset": "openbookqa",
                "kv_quant_scheme": "fp16",
                "kv_cache_proportion": 1.0,
                "kv_cache_order_mode": "front",
                "base_model": "Qwen/Qwen3-0.6B",
                "avg_input_length": 256.0,
                "bytes_per_sequence": 1.0e6,
                "bytes_per_sequence_mib": 1.0e6 / (1024.0 * 1024.0),
                "accuracy": 0.60,
                "run_root": "demo",
            },
            {
                "run_tag": "demo_int8_p50_front",
                "dataset": "openbookqa",
                "kv_quant_scheme": "int8",
                "kv_cache_proportion": 0.5,
                "kv_cache_order_mode": "front",
                "base_model": "Qwen/Qwen3-0.6B",
                "avg_input_length": 256.0,
                "bytes_per_sequence": 2.5e5,
                "bytes_per_sequence_mib": 2.5e5 / (1024.0 * 1024.0),
                "accuracy": 0.50,
                "run_root": "demo",
            },
        ]
        write_csv(rows, out_dir / "budget_curve.csv")
        if not args.no_plot:
            plot_budget_curve(rows, out_dir)
        print(f"Demo outputs in {out_dir}")
        return

    roots = [Path(p) for p in (args.runs_root or ["quantization/data"])]
    rows = []
    for root in roots:
        if not root.exists():
            print(f"Skipping missing runs root: {root}")
            continue
        for run_root in iter_run_roots(root):
            rows.extend(parse_run(run_root))

    if not rows:
        die("No runs found with results. Provide --runs-root after GPU evals.")

    model_stats_cache = {}
    enriched = []
    for row in rows:
        avg_input_length = row["avg_input_length"]
        if avg_input_length is None:
            if args.assume_input_length is None:
                print(f"Skipping {row['run_tag']} ({row['dataset']}): missing length stats")
                continue
            avg_input_length = args.assume_input_length
            row["avg_input_length"] = avg_input_length
        model_name = row["base_model"]
        if model_name not in model_stats_cache:
            model_stats_cache[model_name] = load_model_stats(model_name, override=None)
        model_stats = model_stats_cache[model_name]
        bits = 16 if row["kv_quant_scheme"] == "fp16" else (8 if row["kv_quant_scheme"] == "int8" else 4)
        bpt = bytes_per_token(model_stats, bits)
        bytes_seq = avg_input_length * row["kv_cache_proportion"] * bpt
        row["bytes_per_sequence"] = bytes_seq
        row["bytes_per_sequence_mib"] = bytes_seq / (1024.0 * 1024.0)
        enriched.append(row)

    write_csv(enriched, out_dir / "budget_curve.csv")
    if not args.no_plot:
        plot_budget_curve(enriched, out_dir)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
