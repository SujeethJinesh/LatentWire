"""
Factorial ablation sweep: run calibrate + evaluate for every combination of
the main paper axes on one model pair and one eval file.

Results are written as a JSONL file, one line per configuration:

    {"rotation": "orthogonal", "alignment": "procrustes", "bits": 4,
     "whitening": false, "layer_pairing": "interp", "selection_ratio": 1.0,
     "protocol": "fused", "source_reasoning_mode": "brief_analysis",
     "gate_mode": "fixed", "gate_value": 0.5,
     "target_alone": 0.52, "text_to_text": 0.58, "rotalign_kv": 0.63}

This is the main workhorse for the paper's component study (method.md §5.4).
Intended to be run once per model pair.

Usage:
    python scripts/ablation_sweep.py \
        --source-model Qwen/Qwen2.5-0.5B-Instruct \
        --target-model Qwen/Qwen3-0.6B \
        --calibration-file data/calibration.txt \
        --eval-file data/mcq.jsonl \
        --output results/qwen25_to_qwen3.jsonl

Repeat the same sweep for the agreed small-model matrix once the control pair
is positive:
    - Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-0.8B
    - Qwen/Qwen2.5-0.5B-Instruct -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    - Qwen/Qwen2.5-0.5B-Instruct -> google/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import subprocess
import sys
import time
import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", required=True)
    p.add_argument("--target-model", required=True)
    p.add_argument("--calibration-file", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--output", required=True, help="JSONL results file")
    p.add_argument("--checkpoint-dir", default="checkpoints/sweep")
    p.add_argument(
        "--rotations",
        nargs="+",
        default=["orthogonal", "hadamard"],
        choices=["identity", "orthogonal", "hadamard", "dct"],
    )
    p.add_argument(
        "--alignments",
        nargs="+",
        default=["auto", "ridge", "cca"],
        choices=[
            "auto",
            "identity",
            "procrustes",
            "procrustes_rand",
            "ridge",
            "cca",
            "reduced_rank",
            "grouped_auto",
            "grouped_identity",
            "grouped_procrustes",
            "grouped_procrustes_rand",
            "grouped_ridge",
            "grouped_cca",
            "grouped_reduced_rank",
        ],
    )
    p.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 8])
    p.add_argument("--whiten", nargs="+", default=["off", "on"], choices=["off", "on"])
    p.add_argument("--layer-pairings", nargs="+", default=["interp"], choices=["interp", "cka", "reverse", "shifted", "random"])
    p.add_argument("--selection-ratios", nargs="+", type=float, default=[1.0])
    p.add_argument(
        "--head-selection-topks",
        nargs="+",
        type=int,
        default=[0],
        help="Optional top-k target head-groups to transmit during calibration.",
    )
    p.add_argument(
        "--head-selection-ratios",
        nargs="+",
        type=float,
        default=[1.0],
        help="Fraction of target head-groups to transmit during calibration.",
    )
    p.add_argument(
        "--head-selection-metrics",
        nargs="+",
        default=["mean_cosine_similarity"],
        choices=["mean_cosine_similarity", "negative_error"],
        help="Score used to rank target head-groups before selecting top-k or ratio.",
    )
    p.add_argument(
        "--pre-quant-ranks",
        nargs="+",
        type=int,
        default=[0],
        help="Optional low-rank target-space filter rank before quantization; 0 disables it.",
    )
    p.add_argument(
        "--pre-quant-shrinkages",
        nargs="+",
        type=float,
        default=[0.0],
        help="Shrinkage strength for the pre-quant low-rank filter.",
    )
    p.add_argument(
        "--quantization-corrections",
        nargs="+",
        default=["none"],
        choices=["none", "affine"],
        help="Optional decoder-side correction after quantize/dequantize.",
    )
    p.add_argument("--rotation-seeds", nargs="+", type=int, default=[0])
    p.add_argument(
        "--source-reasoning-modes",
        nargs="+",
        default=["brief_analysis"],
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        help="Source prompt format used for text-to-text and source KV capture",
    )
    p.add_argument(
        "--protocols",
        nargs="+",
        default=["fused"],
        choices=["translated_only", "fused", "text_kv_hybrid"],
    )
    p.add_argument(
        "--kv-transports",
        nargs="+",
        default=["both"],
        choices=["both", "k_only", "v_only"],
        help="Transmit both translated tensors or ablate to keys-only / values-only transport.",
    )
    p.add_argument(
        "--gate-mode",
        choices=["checkpoint", "fixed", "sweep"],
        default="fixed",
    )
    p.add_argument("--gate-values", nargs="+", type=float, default=[0.5])
    p.add_argument("--full-precision-anchor", action="store_true")
    p.add_argument(
        "--source-kv-controls",
        nargs="+",
        default=["real"],
        choices=["real", "zero", "random", "shuffle_positions"],
        help="Source KV negative controls evaluated before translation.",
    )
    p.add_argument(
        "--quantization-controls",
        nargs="+",
        default=["real"],
        choices=["real", "matched_noise"],
        help="Use true quantization or matched-noise control in the quantized path.",
    )
    p.add_argument(
        "--translated-kv-controls",
        nargs="+",
        default=["real"],
        choices=["real", "zero", "random", "shuffle_positions"],
        help="Target-space translated KV controls evaluated after translation.",
    )
    p.add_argument(
        "--fusion-rules",
        nargs="+",
        default=["static"],
        choices=["static", "cosine", "cosine_shifted", "js_shrinkage", "kalman"],
        help="Runtime fusion rule used when combining target and translated KV.",
    )
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", default="float32")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    """Run a subprocess, return its stdout. Raises on nonzero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Subprocess failed")
    return result.stdout


def parse_accuracies(evaluate_output: str) -> dict[str, float]:
    """Extract the '=== Summary ===' block from scripts/evaluate.py stdout."""
    results: dict[str, float] = {}
    in_summary = False
    for line in evaluate_output.splitlines():
        if "=== Summary ===" in line:
            in_summary = True
            continue
        if in_summary and ":" in line:
            try:
                name, value = line.split(":", 1)
                results[name.strip()] = float(value.strip())
            except ValueError:
                pass
    return results


def main() -> None:
    args = parse_args()
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    calibrate = str(repo_root / "scripts" / "calibrate.py")
    evaluate = str(repo_root / "scripts" / "evaluate.py")

    quantize_modes = ["quantized"]
    if getattr(args, "full_precision_anchor", False):
        quantize_modes.append("full_precision")
    combos = list(
        itertools.product(
            args.rotations,
            args.alignments,
            args.bits,
            args.whiten,
            getattr(args, "layer_pairings", ["interp"]),
            getattr(args, "selection_ratios", [1.0]),
            getattr(args, "head_selection_topks", []),
            getattr(args, "head_selection_ratios", [1.0]),
            getattr(args, "head_selection_metrics", ["mean_cosine_similarity"]),
            getattr(args, "pre_quant_ranks", [0]),
            getattr(args, "pre_quant_shrinkages", [0.0]),
            getattr(args, "quantization_corrections", ["none"]),
            getattr(args, "rotation_seeds", [0]),
            getattr(args, "source_reasoning_modes", ["brief_analysis"]),
            getattr(args, "protocols", ["fused"]),
            getattr(args, "kv_transports", ["both"]),
            quantize_modes,
            getattr(args, "source_kv_controls", ["real"]),
            getattr(args, "quantization_controls", ["real"]),
            getattr(args, "translated_kv_controls", ["real"]),
            getattr(args, "fusion_rules", ["static"]),
        )
    )
    print(f"Running {len(combos)} configurations...")

    with open(out_path, "w") as f_out:
        for i, (
            rotation,
            alignment,
            bits,
            whiten,
            layer_pairing,
            selection_ratio,
            head_selection_topk,
            head_selection_ratio,
            head_selection_metric,
            pre_quant_rank,
            pre_quant_shrinkage,
            quantization_correction,
            seed,
            source_reasoning_mode,
            protocol,
            kv_transport,
            quantize_mode,
            source_kv_control,
            quantization_control,
            translated_kv_control,
            fusion_rule,
        ) in enumerate(combos):
            tag = (
                f"rot{rotation}_align{alignment}_bits{bits}_w{whiten}"
                f"_pair{layer_pairing}_sel{selection_ratio}_seed{seed}"
                f"_headtopk{head_selection_topk if head_selection_topk else 'none'}"
                f"_headsel{head_selection_ratio}_headmetric{head_selection_metric}"
                f"_prerank{pre_quant_rank}_preshrink{pre_quant_shrinkage}"
                f"_qcorr{quantization_correction}"
                f"_reason{source_reasoning_mode}_proto{protocol}"
                f"_kv{kv_transport}"
                f"_q{quantize_mode}"
                f"_srcctrl{source_kv_control}_qctrl{quantization_control}"
                f"_tgtctrl{translated_kv_control}"
                f"_fuse{fusion_rule}"
            )
            ckpt = ckpt_dir / f"{tag}.pt"
            start = time.time()

            # 1. Calibrate
            cal_cmd = [
                sys.executable, calibrate,
                "--source-model", args.source_model,
                "--target-model", args.target_model,
                "--calibration-file", args.calibration_file,
                "--output", str(ckpt),
                "--bits", str(bits),
                "--rotation", rotation,
                "--alignment", alignment,
                "--layer-pairing", layer_pairing,
                "--layer-selection-ratio", str(selection_ratio),
                "--seed", str(seed),
                "--device", args.device,
                "--dtype", args.dtype,
                "--source-reasoning-mode", source_reasoning_mode,
            ]
            if head_selection_topk > 0:
                cal_cmd.extend(["--head-selection-topk", str(head_selection_topk)])
            if head_selection_ratio < 1.0:
                cal_cmd.extend(["--head-selection-ratio", str(head_selection_ratio)])
            if head_selection_metric != "mean_cosine_similarity":
                cal_cmd.extend(["--head-selection-metric", head_selection_metric])
            if pre_quant_rank > 0:
                cal_cmd.extend(["--pre-quant-rank", str(pre_quant_rank)])
            if pre_quant_shrinkage > 0.0:
                cal_cmd.extend(["--pre-quant-shrinkage", str(pre_quant_shrinkage)])
            if quantization_correction != "none":
                cal_cmd.extend(["--quantization-correction", quantization_correction])
            if whiten == "on":
                cal_cmd.append("--whitening")

            try:
                run_cmd(cal_cmd)
            except RuntimeError:
                print(f"[{i+1}/{len(combos)}] {tag}: calibration FAILED, skipping")
                continue

            # 2. Evaluate
            eval_cmd = [
                sys.executable, evaluate,
                "--translator", str(ckpt),
                "--source-model", args.source_model,
                "--target-model", args.target_model,
                "--eval-file", args.eval_file,
                "--methods", "target", "t2t", "rotalign",
                "--task-type", "auto",
                "--gate-mode", getattr(args, "gate_mode", "fixed"),
                "--device", args.device,
                "--dtype", args.dtype,
                "--source-reasoning-mode", source_reasoning_mode,
            ]
            if protocol == "translated_only":
                eval_cmd[eval_cmd.index("rotalign")] = "rotalign_translated"
            elif protocol == "text_kv_hybrid":
                eval_cmd[eval_cmd.index("rotalign")] = "rotalign_text_kv"
            if kv_transport != "both":
                eval_cmd.extend(["--kv-transport", kv_transport])
            gate_values = getattr(args, "gate_values", [0.5])
            if getattr(args, "gate_mode", "fixed") in {"fixed", "sweep"}:
                eval_cmd.extend(["--gate-values", *[str(v) for v in gate_values]])
                if getattr(args, "gate_mode", "fixed") == "fixed":
                    eval_cmd.extend(["--fixed-gate", str(gate_values[0])])
            if quantize_mode == "full_precision":
                eval_cmd.append("--no-quantize")
            if source_kv_control != "real":
                eval_cmd.extend(["--source-kv-control", source_kv_control])
            if quantization_control != "real":
                eval_cmd.extend(["--quantization-control", quantization_control])
            if translated_kv_control != "real":
                eval_cmd.extend(["--translated-kv-control", translated_kv_control])
            if fusion_rule != "static":
                eval_cmd.extend(["--fusion-rule", fusion_rule])
            try:
                output = run_cmd(eval_cmd)
            except RuntimeError:
                print(f"[{i+1}/{len(combos)}] {tag}: evaluation FAILED, skipping")
                continue

            accs = parse_accuracies(output)
            elapsed = time.time() - start

            record = {
                "rotation": rotation,
                "alignment": alignment,
                "bits": bits,
                "whitening": whiten == "on",
                "layer_pairing": layer_pairing,
                "selection_ratio": selection_ratio,
                "head_selection_topk": None if head_selection_topk <= 0 else head_selection_topk,
                "head_selection_ratio": head_selection_ratio,
                "head_selection_metric": head_selection_metric,
                "pre_quant_rank": None if pre_quant_rank <= 0 else pre_quant_rank,
                "pre_quant_shrinkage": pre_quant_shrinkage,
                "quantization_correction": quantization_correction,
                "rotation_seed": seed,
                "source_reasoning_mode": source_reasoning_mode,
                "protocol": protocol,
                "kv_transport": kv_transport,
                "quantize_mode": quantize_mode,
                "source_kv_control": source_kv_control,
                "quantization_control": quantization_control,
                "translated_kv_control": translated_kv_control,
                "fusion_rule": fusion_rule,
                "gate_mode": getattr(args, "gate_mode", "fixed"),
                "gate_values": getattr(args, "gate_values", [0.5]),
                "elapsed_sec": round(elapsed, 1),
                **accs,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            print(
                f"[{i+1}/{len(combos)}] {tag}: "
                f"rotalign={accs.get('rotalign_kv', float('nan')):.3f} "
                f"t2t={accs.get('text_to_text', float('nan')):.3f} "
                f"target={accs.get('target_alone', float('nan')):.3f} "
                f"({elapsed:.0f}s)"
            )

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
