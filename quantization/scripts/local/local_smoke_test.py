#!/usr/bin/env python3
"""Local Mac smoke test for C2C (single prompt, tiny generation).

This script is intentionally minimal and CPU/MPS-friendly. It does NOT run
full evaluation; it only verifies that model loading + projector fusion works
end-to-end on a single prompt.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local import path for rosetta
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "quantization" / "C2C"))

from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.utils.evaluate import set_default_chat_template


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    parser = argparse.ArgumentParser(description="Local C2C smoke test (Mac-friendly).")
    parser.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint-dir", default=None, help="Local checkpoint dir (projector_*.pt/json)")
    parser.add_argument("--run-tag", default=None, help="Name for local run output folder")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory for artifacts")
    parser.add_argument("--hf-cache", default=None, help="Optional HF cache dir")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache, "transformers")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_tag = args.run_tag or datetime.now().strftime("local_smoke_%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "data" / "local_smoke_tests" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure C2C submodule is present
    c2c_root = REPO_ROOT / "quantization" / "C2C"
    if not c2c_root.exists():
        raise SystemExit(f"C2C submodule not found at {c2c_root}")

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        # Download minimal projector checkpoint for qwen3_0.6b + qwen2.5_0.5b
        snapshot = snapshot_download(
            repo_id="nics-efc/C2C_Fuser",
            allow_patterns=["qwen3_0.6b+qwen2.5_0.5b_Fuser/*"],
            local_dir_use_symlinks=False,
        )
        ckpt_dir = str(Path(snapshot) / "qwen3_0.6b+qwen2.5_0.5b_Fuser" / "final")

    # Load tokenizer + models
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    set_default_chat_template(tokenizer, args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map={"": device}
    ).eval()
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype, device_map={"": device}
    ).eval()

    # Load projectors
    ckpt_path = Path(ckpt_dir)
    projector_list = []
    for proj_json in sorted(ckpt_path.glob("projector_*.json")):
        proj = load_projector(str(proj_json))
        proj = proj.to(device)
        pt_path = proj_json.with_suffix(".pt")
        if pt_path.exists():
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)

    rosetta = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(device).eval()

    # Load projector config mapping
    proj_cfg_path = ckpt_path / "projector_config.json"
    rosetta.load_projector_config(str(proj_cfg_path))

    # Build a tiny prompt
    prompt = [{"role": "user", "content": "Answer in one word: What is 2+2?"}]
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs["input_ids"].shape[1] - 1, 1).unsqueeze(0).to(device)
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
    kv_cache_index = [instruction_index, label_index]

    with torch.no_grad():
        outputs = rosetta.generate(
            **inputs,
            kv_cache_index=kv_cache_index,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("C2C smoke test output:")
    print(text)

    meta = {
        "device": str(device),
        "dtype": str(dtype),
        "base_model": args.base_model,
        "teacher_model": args.teacher_model,
        "checkpoint_dir": ckpt_dir,
        "max_new_tokens": args.max_new_tokens,
        "output_dir": str(output_dir),
    }
    print("Run metadata:\n" + json.dumps(meta, indent=2))

    (output_dir / "output.txt").write_text(text + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
