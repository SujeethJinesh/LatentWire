#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_conda_exe():
    return os.environ.get("CONDA_EXE") or shutil.which("conda")


def find_conda_env_prefix(conda_exe, env_name):
    try:
        res = subprocess.run(
            [conda_exe, "env", "list", "--json"], capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        for env_path in data.get("envs", []):
            if Path(env_path).name == env_name:
                return env_path
    except Exception:
        return None
    return None


def ensure_env(env_name):
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env == env_name:
        return
    if "--no-reexec" in sys.argv:
        die(f"Not running inside conda env '{env_name}'. Activate it and retry.")
    conda_exe = find_conda_exe()
    if conda_exe is None:
        die("conda not found in PATH. Load your conda module and retry.")
    res = subprocess.run([conda_exe, "env", "list", "--json"], capture_output=True, text=True)
    if res.returncode != 0:
        die("Failed to list conda environments.")
    try:
        data = json.loads(res.stdout)
    except Exception:
        data = {}
    envs = data.get("envs", [])
    if not any(Path(p).name == env_name for p in envs):
        print(f"Conda env '{env_name}' not found. Creating it now...")
        subprocess.check_call([conda_exe, "create", "-n", env_name, "python=3.10", "-y"])
    env_prefix = find_conda_env_prefix(conda_exe, env_name)
    python_exe = None
    if env_prefix:
        candidate = Path(env_prefix) / "bin" / "python"
        if candidate.exists():
            python_exe = str(candidate)
    script_path = Path(__file__).resolve()
    forwarded = [arg for arg in sys.argv[1:] if arg != "--no-reexec"]
    python_cmd = python_exe or "python"
    cmd = [conda_exe, "run", "-n", env_name, python_cmd, str(script_path)] + forwarded + ["--no-reexec"]
    print("Re-running inside conda env:", " ".join(cmd))
    subprocess.check_call(cmd)
    sys.exit(0)


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


def _normalize_scheme(scheme):
    if scheme is None:
        return "fp16"
    if isinstance(scheme, str):
        return scheme.lower()
    return str(scheme).lower()


def _scheme_bits(scheme):
    scheme = _normalize_scheme(scheme)
    if scheme in ("int4", "nf4"):
        return 4
    if scheme in ("int8", "fp8"):
        return 8
    if scheme in ("fp16", "bf16", "float16", "bfloat16"):
        return 16
    if scheme in ("fp32", "float32"):
        return 32
    if scheme in ("none", "no"):
        return 16
    return 16


def _apply_schedule_overrides(schemes, overrides):
    for override in overrides or []:
        scheme = _normalize_scheme(override.get("scheme"))
        if not scheme:
            continue
        if override.get("layers") is not None:
            for idx in override.get("layers") or []:
                try:
                    layer_idx = int(idx)
                except Exception:
                    continue
                if 0 <= layer_idx < len(schemes):
                    schemes[layer_idx] = scheme
        elif override.get("range") is not None:
            r = override.get("range")
            if isinstance(r, (list, tuple)) and len(r) == 2:
                try:
                    start, end = int(r[0]), int(r[1])
                except Exception:
                    continue
                start = max(start, 0)
                end = min(end, len(schemes) - 1)
                for layer_idx in range(start, end + 1):
                    schemes[layer_idx] = scheme


def quant_bits_from_config(kv_quant_config, num_layers=None):
    if not kv_quant_config or not kv_quant_config.get("enabled", False):
        return 16, "fp16"
    scheme = _normalize_scheme(kv_quant_config.get("scheme", "int8"))
    schedule = kv_quant_config.get("layer_schedule") or {}
    if schedule and num_layers:
        default_scheme = _normalize_scheme(schedule.get("default", scheme))
        schemes = [default_scheme for _ in range(int(num_layers))]
        _apply_schedule_overrides(schemes, schedule.get("overrides", []))
        avg_bits = sum(_scheme_bits(s) for s in schemes) / float(len(schemes))
        unique = []
        for s in schemes:
            if s not in unique:
                unique.append(s)
        label = f"mixed({','.join(unique)})" if len(unique) > 1 else unique[0]
        return avg_bits, label
    return _scheme_bits(scheme), scheme


def normalize_model_stats(stats):
    if not isinstance(stats, dict):
        return None
    hidden_size = stats.get("hidden_size")
    num_layers = stats.get("num_layers")
    num_heads = stats.get("num_heads")
    num_kv_heads = stats.get("num_kv_heads")
    head_dim = stats.get("head_dim")
    if head_dim is None and hidden_size is not None and num_heads:
        head_dim = int(hidden_size) // int(num_heads)
    required = [hidden_size, num_layers, num_heads, num_kv_heads, head_dim]
    if any(v is None for v in required):
        return None
    return {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }


def load_model_stats(model_name, override, local_files_only=False):
    if override:
        normalized = normalize_model_stats(override)
        if normalized:
            return normalized
        die(f"Invalid model stats override for {model_name}")
    try:
        from transformers import AutoConfig
    except Exception as exc:
        die(f"transformers not available to load model config: {exc}")
    cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
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


def load_model_stats_overrides(paths):
    overrides = {}
    for path in paths or []:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            die(f"Model stats file must be a JSON object: {path}")
        for model_name, stats in data.items():
            normalized = normalize_model_stats(stats)
            if not normalized:
                die(f"Invalid stats for {model_name} in {path}")
            overrides[model_name] = normalized
    return overrides


def read_manifest_model_stats(run_root):
    manifest_dir = run_root / "manifests"
    if not manifest_dir.exists():
        return None
    for name in ("step_1_manifest.json", "step_0_checkpoint_manifest.json", "run_manifest.json"):
        path = manifest_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("base_model_stats", "model_stats"):
            stats = data.get(key)
            normalized = normalize_model_stats(stats)
            if normalized:
                return normalized
    return None


def bytes_per_token(model_stats, bits_per_elem):
    bytes_per_elem = bits_per_elem / 8.0
    return 2.0 * model_stats["num_layers"] * model_stats["num_kv_heads"] * model_stats["head_dim"] * bytes_per_elem


def iter_run_roots(root):
    for sub in ("step_0_baselines", "step_1_kv_ptq", "step_8_selective_transfer"):
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
    manifest_stats = read_manifest_model_stats(run_root)
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
        kv_transfer_config = model_cfg.get("kv_transfer_config") or manifest.get("kv_transfer_config")
        num_layers = manifest_stats["num_layers"] if isinstance(manifest_stats, dict) else None
        bits, scheme = quant_bits_from_config(kv_quant_config, num_layers=num_layers)
        eval_cfg = cfg.get("eval", {})
        proportion = eval_cfg.get("kv_cache_proportion", manifest.get("kv_cache_proportion", 1.0))
        order_mode = eval_cfg.get("kv_cache_order_mode", manifest.get("kv_cache_order_mode", "front"))
        token_select_mode = None
        token_select_proportion = None
        token_select_scope = None
        sparse_fuse = None
        index_dtype_bytes = None
        include_scale_overhead = None
        if isinstance(kv_transfer_config, dict) and kv_transfer_config.get("enabled", False):
            token_select_mode = kv_transfer_config.get("token_select_mode")
            token_select_proportion = kv_transfer_config.get("token_select_proportion")
            token_select_scope = kv_transfer_config.get("token_select_scope")
            sparse_fuse = kv_transfer_config.get("sparse_fuse")
            index_dtype_bytes = kv_transfer_config.get("index_dtype_bytes")
            include_scale_overhead = kv_transfer_config.get("include_scale_overhead")
        rows.append(
            {
                "run_root": str(run_root),
                "run_tag": run_root.name,
                "dataset": dataset_dir.name,
                "base_model": base_model,
                "kv_quant_scheme": scheme,
                "kv_quant_bits": bits,
                "kv_cache_proportion": float(proportion),
                "kv_cache_order_mode": order_mode,
                "token_select_mode": token_select_mode,
                "token_select_proportion": token_select_proportion,
                "token_select_scope": token_select_scope,
                "sparse_fuse": sparse_fuse,
                "index_dtype_bytes": index_dtype_bytes,
                "include_scale_overhead": include_scale_overhead,
                "accuracy": safe_float(summary.get("overall_accuracy")),
                "avg_input_length": extract_avg_input_length(summary),
                "model_stats": manifest_stats,
            }
        )
    return rows


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "dataset",
        "kv_quant_scheme",
        "kv_quant_bits",
        "kv_cache_proportion",
        "kv_cache_order_mode",
        "token_select_mode",
        "token_select_proportion",
        "token_select_scope",
        "sparse_fuse",
        "index_dtype_bytes",
        "include_scale_overhead",
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


def plot_budget_curve(rows, out_dir, install_deps=False):
    try:
        import numpy as np
    except Exception:
        if not install_deps:
            print("Skipping plot (numpy missing). Install numpy<2 or pass --install-plot-deps.")
            return
        import subprocess
        print("numpy not available; installing a compatible version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2"])
        import numpy as np

    try:
        import matplotlib.pyplot as plt
    except Exception:
        if not install_deps:
            print("Skipping plot (matplotlib missing). Install matplotlib or pass --install-plot-deps.")
            return
        import subprocess
        print("matplotlib not found; installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in rows})
    scheme_colors = {"fp16": "#111111", "int8": "#1f77b4", "int4": "#ff7f0e", "nf4": "#d62728", "fp8": "#9467bd"}
    order_styles = {"front": ("-", "o"), "back": ("--", "s")}
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        # Group by scheme + order for connected curves
        groups = {}
        for row in dataset_rows:
            scheme = row["kv_quant_scheme"]
            order = row["kv_cache_order_mode"]
            key = (scheme, order)
            groups.setdefault(key, []).append(row)

        for (scheme, order), group in groups.items():
            group = sorted(group, key=lambda r: r["bytes_per_sequence"])
            x = [r["bytes_per_sequence_mib"] for r in group]
            y = [r["accuracy"] for r in group]
            linestyle, marker = order_styles.get(order, ("-", "o"))
            ax.plot(
                x,
                y,
                linestyle=linestyle,
                marker=marker,
                color=scheme_colors.get(scheme, "#999999"),
                label=f"{scheme.upper()} / {order}",
            )
            # Annotate cache proportion for readability
            for row in group:
                prop = row.get("kv_cache_proportion")
                if prop is None:
                    continue
                ax.annotate(
                    f"{prop:.2f}",
                    (row["bytes_per_sequence_mib"], row["accuracy"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=scheme_colors.get(scheme, "#666666"),
                )

        ax.set_xlabel("MiB per sequence (approx)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs Communication Budget ({dataset})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Scheme / Order", loc="best")
        # Log scale helps separate FP16/INT8/INT4 points if span is wide.
        try:
            xs = [r["bytes_per_sequence_mib"] for r in dataset_rows if r["bytes_per_sequence_mib"]]
            if xs and max(xs) / min(xs) >= 4:
                ax.set_xscale("log")
        except Exception:
            pass
        out_path = out_dir / f"budget_curve_{dataset}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy vs bytes curves for C2C runs.")
    parser.add_argument("--env", default="rosetta", help="Conda env name to use")
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
    parser.add_argument(
        "--model-stats-file",
        action="append",
        default=[],
        help="JSON file mapping model names to stats (hidden_size/num_layers/num_heads/num_kv_heads/head_dim)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating PNG plots")
    parser.add_argument("--install-plot-deps", action="store_true", help="Install numpy/matplotlib if missing")
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic CSV/plot for validation")
    parser.add_argument("--no-reexec", action="store_true", help="Internal flag to avoid re-exec loops")
    args = parser.parse_args()

    ensure_env(args.env)

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
            plot_budget_curve(rows, out_dir, install_deps=args.install_plot_deps)
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
    overrides = load_model_stats_overrides(args.model_stats_file)
    row_overrides = {}
    for row in rows:
        stats = row.get("model_stats")
        model_name = row.get("base_model")
        if stats and model_name and model_name not in row_overrides:
            row_overrides[model_name] = stats
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")
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
            override = row_overrides.get(model_name) or overrides.get(model_name)
            model_stats_cache[model_name] = load_model_stats(
                model_name, override=override, local_files_only=local_only
            )
        model_stats = model_stats_cache[model_name]
        bits = row.get("kv_quant_bits")
        if bits is None:
            bits = 16 if row["kv_quant_scheme"] == "fp16" else (8 if row["kv_quant_scheme"] == "int8" else 4)
        bpt = bytes_per_token(model_stats, bits)
        effective_prop = row["kv_cache_proportion"]
        if row.get("token_select_proportion") is not None:
            try:
                effective_prop = effective_prop * float(row["token_select_proportion"])
            except Exception:
                pass
        bytes_seq = avg_input_length * effective_prop * bpt
        index_bytes = 0.0
        if row.get("index_dtype_bytes") is not None:
            try:
                index_bytes = avg_input_length * effective_prop * float(row["index_dtype_bytes"])
            except Exception:
                index_bytes = 0.0
        bytes_seq += index_bytes
        if row.get("include_scale_overhead") and model_stats:
            # Approximate per-head scale overhead: 2 scales (K/V) per layer per head (float32)
            overhead = 2 * model_stats["num_layers"] * model_stats["num_kv_heads"] * 4
            bytes_seq += overhead
        row["bytes_per_sequence"] = bytes_seq
        row["bytes_per_sequence_mib"] = bytes_seq / (1024.0 * 1024.0)
        enriched.append(row)

    write_csv(enriched, out_dir / "budget_curve.csv")
    if not args.no_plot:
        plot_budget_curve(enriched, out_dir, install_deps=args.install_plot_deps)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
