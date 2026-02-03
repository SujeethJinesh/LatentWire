#!/usr/bin/env python3
import argparse
import json
import os
import sys


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Check HF model access and print key shapes.")
    parser.add_argument("model_id", help="Model identifier (HF repo or local path)")
    parser.add_argument("--local-only", action="store_true", help="Use local files only")
    args = parser.parse_args()

    try:
        from transformers import AutoConfig
    except Exception as exc:
        die(f"transformers not available: {exc}")

    try:
        cfg = AutoConfig.from_pretrained(args.model_id, local_files_only=args.local_only)
    except Exception as exc:
        die(f"Failed to load model config for {args.model_id}: {exc}")

    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
    max_positions = getattr(cfg, "max_position_embeddings", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    head_dim = None
    if hidden_size and num_heads:
        head_dim = int(hidden_size) // int(num_heads)

    info = {
        "model_id": args.model_id,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_positions,
        "rope_scaling": rope_scaling,
    }
    print(json.dumps(info, indent=2))

    if not all([hidden_size, num_layers, num_heads, num_kv_heads]):
        die("Missing required model stats; check model access or config.")


if __name__ == "__main__":
    main()
