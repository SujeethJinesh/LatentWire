#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print("$ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description="Run Step 11 KVWire validation (local smoke or GPU).")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--mode", choices=["local", "gpu"], default="local")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--eval-datasets", default="openbookqa,ai2-arc")
    parser.add_argument("--kv-quant-scheme", choices=["int8", "int4"], default="int8")
    parser.add_argument("--wire-quant-mode", choices=["int8", "int4"], default="int8")
    parser.add_argument("--wire-apply-pack", dest="wire_apply_pack", action="store_true", help="Apply KVWire pack/unpack.")
    parser.add_argument("--wire-no-apply-pack", dest="wire_apply_pack", action="store_false", help="Disable KVWire pack/unpack.")
    parser.set_defaults(wire_apply_pack=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    script_root = project_root / "quantization" / "scripts"
    if args.run_tag is None:
        args.run_tag = time.strftime("step11_kvwire_%Y%m%d_%H%M%S")

    # Always run a tiny KVWire equivalence check.
    run_cmd([sys.executable, str(script_root / "debug_kvwire_equivalence.py"), "--quant", args.wire_quant_mode])

    if args.mode == "local":
        cmd = [
            sys.executable,
            str(script_root / "run_step8_selective_transfer.py"),
            "--mode", "local",
            "--kv-quant-scheme", args.kv_quant_scheme,
            "--kv-select-mode", "vnorm_topk",
            "--kv-select-proportion", "0.25",
            "--wire-format", "kvwire_v1",
            "--wire-quant-mode", args.wire_quant_mode,
            "--wire-record-per-sample",
            "--wire-sample-limit", "10",
            "--run-tag", args.run_tag,
        ]
        if args.wire_apply_pack:
            cmd.append("--wire-apply-pack")
        run_cmd(cmd)
    else:
        cmd = [
            sys.executable,
            str(script_root / "run_step8_selective_transfer.py"),
            "--mode", "gpu",
            "--kv-quant-scheme", args.kv_quant_scheme,
            "--kv-select-mode", "vnorm_topk",
            "--kv-select-proportion", "0.25",
            "--wire-format", "kvwire_v1",
            "--wire-quant-mode", args.wire_quant_mode,
            "--wire-record-per-sample",
            "--wire-sample-limit", "10",
            "--eval-datasets", args.eval_datasets,
            "--run-tag", args.run_tag,
        ]
        if args.wire_apply_pack:
            cmd.append("--wire-apply-pack")
        run_cmd(cmd)


if __name__ == "__main__":
    main()
