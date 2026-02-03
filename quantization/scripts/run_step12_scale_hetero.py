#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None, env=None, dry_run=False):
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=cwd, env=env)


def patch_manifest(project_root: Path, run_tag: str, pair_id: str, context_len: str):
    manifest_path = project_root / "quantization" / "data" / "step_8_selective_transfer" / run_tag / "manifests" / "step_8_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        manifest = {}
    manifest["pair_id"] = pair_id
    manifest["context_len_bucket"] = context_len
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_json(path: Path):
    return json.loads(path.read_text())


def model_stats(model_id: str, local_only=False):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, local_files_only=local_only)
    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
    if not all([hidden_size, num_layers, num_heads, num_kv_heads]):
        raise RuntimeError(f"Missing model stats for {model_id}")
    head_dim = int(hidden_size) // int(num_heads)
    return {
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }


def bytes_per_token(stats, bytes_per_element=1.0):
    return 2 * stats["num_kv_heads"] * stats["head_dim"] * bytes_per_element


def main():
    parser = argparse.ArgumentParser(description="Run M12 scale+hetero grid.")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--grid", default="quantization/configs/grids/scale_hetero_grid.json")
    parser.add_argument("--mode", choices=["gpu", "local"], default="gpu")
    parser.add_argument("--exec-mode", choices=["simultaneous", "sequential"], default="simultaneous")
    parser.add_argument("--method", choices=["m9", "m10"], default="m10")
    parser.add_argument("--run-tag-prefix", default=None)
    parser.add_argument("--smoke-first", dest="smoke_first", action="store_true", help="Run a smoke eval (limit) before full runs.")
    parser.add_argument("--no-smoke-first", dest="smoke_first", action="store_false", help="Disable smoke pre-run gate.")
    parser.set_defaults(smoke_first=True)
    parser.add_argument("--smoke-limit", type=int, default=50, help="Sample limit for smoke runs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    grid = load_json(project_root / args.grid)

    pairs = grid.get("pair_id") or grid.get("pairs") or []
    tasks = grid.get("task") or grid.get("tasks") or ["openbookqa", "ai2-arc"]
    context_lens = grid.get("context_len") or ["baseline"]
    budgets = grid.get("budget") or []
    select_props = grid.get("token_select_proportion") or grid.get("token_select_proportions") or [1.0]
    assumed_seq_len = grid.get("assumed_seq_len", 512)
    kv_quant_scheme = grid.get("kv_quant_scheme", "int8")
    wire_corpus = grid.get("long_context_corpus", "quantization/prompts/longctx_corpus.jsonl")
    long_tokens = grid.get("long_context_tokens", 8192)
    precision_candidates = grid.get("token_precision_candidates", "drop,int4,int8")

    if not pairs:
        raise SystemExit("No pairs specified in grid")

    run_tag_prefix = args.run_tag_prefix or time.strftime("step12_%Y%m%d_%H%M%S")

    for pair_id in pairs:
        pair_cfg_path = project_root / "quantization" / "configs" / "pairs" / f"{pair_id}.json"
        pair_cfg = load_json(pair_cfg_path)
        receiver = pair_cfg["receiver_model"]
        sharer = pair_cfg["sharer_model"]
        alignment_on = bool(pair_cfg.get("alignment", True))
        alignment_strategy = pair_cfg.get("alignment_strategy")

        stats = model_stats(receiver, local_only=False)
        bpt = bytes_per_token(stats, bytes_per_element=1.0)

        for context_len in context_lens:
            long_context = context_len == "long"
            for task in tasks:
                if args.smoke_first:
                    smoke_tag = f"smoke_{run_tag_prefix}_{pair_id}_{context_len}_{task}_{args.method}"
                    smoke_root = project_root / "quantization" / "data" / "step_8_selective_transfer" / smoke_tag
                    if not smoke_root.exists():
                        if args.method == "m9":
                            smoke_prop = select_props[0] if select_props else 1.0
                            cmd = [
                                sys.executable,
                                str(project_root / "quantization" / "scripts" / "run_step8_selective_transfer.py"),
                                "--mode", args.mode,
                                "--run-tag", smoke_tag,
                                "--base-model-override", receiver,
                                "--teacher-model-override", sharer,
                                "--kv-quant-scheme", kv_quant_scheme,
                                "--kv-select-mode", "delta_proj_vnorm_topk",
                                "--kv-select-proportion", str(smoke_prop),
                                "--wire-format", "kvwire_v1",
                                "--wire-apply-pack",
                                "--wire-record-per-sample",
                                "--eval-datasets", task,
                                "--eval-limit", str(args.smoke_limit),
                                "--exec-mode", args.exec_mode,
                            ]
                        else:
                            smoke_budget = budgets[0] if budgets else 0.125
                            if isinstance(smoke_budget, str) and "/" in smoke_budget:
                                num, den = smoke_budget.split("/", 1)
                                smoke_budget = float(num) / float(den)
                            smoke_budget = float(smoke_budget)
                            per_layer_budget = smoke_budget * float(assumed_seq_len) * float(bpt)
                            cmd = [
                                sys.executable,
                                str(project_root / "quantization" / "scripts" / "run_step8_selective_transfer.py"),
                                "--mode", args.mode,
                                "--run-tag", smoke_tag,
                                "--base-model-override", receiver,
                                "--teacher-model-override", sharer,
                                "--kv-quant-scheme", kv_quant_scheme,
                                "--token-precision-mode", "rd_greedy",
                                "--token-precision-candidates", precision_candidates,
                                "--token-precision-budget-bytes", str(per_layer_budget),
                                "--wire-format", "kvwire_v1",
                                "--wire-apply-pack",
                                "--wire-record-per-sample",
                                "--eval-datasets", task,
                                "--eval-limit", str(args.smoke_limit),
                                "--exec-mode", args.exec_mode,
                            ]
                        if alignment_on:
                            cmd.append("--do-alignment")
                        if alignment_strategy:
                            cmd.extend(["--alignment-strategy", alignment_strategy])
                        if long_context:
                            cmd.extend([
                                "--long-context",
                                "--long-context-tokens", str(long_tokens),
                                "--long-context-corpus", wire_corpus,
                            ])
                        run_cmd(cmd, cwd=str(project_root), dry_run=args.dry_run)
                        if not args.dry_run:
                            patch_manifest(project_root, smoke_tag, pair_id, context_len)
                if args.method == "m9":
                    for prop in select_props:
                        tag = f"{run_tag_prefix}_{pair_id}_{context_len}_m9_p{prop}"
                        cmd = [
                            sys.executable,
                            str(project_root / "quantization" / "scripts" / "run_step8_selective_transfer.py"),
                            "--mode", args.mode,
                            "--run-tag", tag,
                            "--base-model-override", receiver,
                            "--teacher-model-override", sharer,
                            "--kv-quant-scheme", kv_quant_scheme,
                            "--kv-select-mode", "delta_proj_vnorm_topk",
                            "--kv-select-proportion", str(prop),
                            "--wire-format", "kvwire_v1",
                            "--wire-apply-pack",
                            "--wire-record-per-sample",
                            "--eval-datasets", task,
                            "--exec-mode", args.exec_mode,
                        ]
                        if alignment_on:
                            cmd.append("--do-alignment")
                        if alignment_strategy:
                            cmd.extend(["--alignment-strategy", alignment_strategy])
                        if long_context:
                            cmd.extend([
                                "--long-context",
                                "--long-context-tokens", str(long_tokens),
                                "--long-context-corpus", wire_corpus,
                            ])
                        run_cmd(cmd, cwd=str(project_root), dry_run=args.dry_run)
                        if not args.dry_run:
                            patch_manifest(project_root, tag, pair_id, context_len)
                else:
                    for budget in budgets:
                        if isinstance(budget, str) and "/" in budget:
                            num, den = budget.split("/", 1)
                            budget = float(num) / float(den)
                        budget = float(budget)
                        per_layer_budget = budget * float(assumed_seq_len) * float(bpt)
                        tag = f"{run_tag_prefix}_{pair_id}_{context_len}_m10_b{budget}"
                        cmd = [
                            sys.executable,
                            str(project_root / "quantization" / "scripts" / "run_step8_selective_transfer.py"),
                            "--mode", args.mode,
                            "--run-tag", tag,
                            "--base-model-override", receiver,
                            "--teacher-model-override", sharer,
                            "--kv-quant-scheme", kv_quant_scheme,
                            "--token-precision-mode", "rd_greedy",
                            "--token-precision-candidates", precision_candidates,
                            "--token-precision-budget-bytes", str(per_layer_budget),
                            "--wire-format", "kvwire_v1",
                            "--wire-apply-pack",
                            "--wire-record-per-sample",
                            "--eval-datasets", task,
                            "--exec-mode", args.exec_mode,
                        ]
                        if alignment_on:
                            cmd.append("--do-alignment")
                        if alignment_strategy:
                            cmd.extend(["--alignment-strategy", alignment_strategy])
                        if long_context:
                            cmd.extend([
                                "--long-context",
                                "--long-context-tokens", str(long_tokens),
                                "--long-context-corpus", wire_corpus,
                            ])
                        run_cmd(cmd, cwd=str(project_root), dry_run=args.dry_run)
                        if not args.dry_run:
                            patch_manifest(project_root, tag, pair_id, context_len)


if __name__ == "__main__":
    main()
