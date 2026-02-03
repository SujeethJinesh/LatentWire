#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from rosetta.utils.evaluate import build_prompt, extract_answer_from_content, set_default_chat_template

from quantization.baselines import apply_payload_to_receiver
from quantization.baselines.text_comm import make_message, count_utf8_bytes


def _load_dataset(name):
    if name == "openbookqa":
        return load_dataset("allenai/openbookqa", split="test")
    if name == "ai2-arc":
        return load_dataset("allenai/ai2_arc", split="test")
    raise ValueError(f"Unsupported dataset: {name}")


def _format_example(example, dataset):
    if dataset == "openbookqa":
        question = example.get("question_stem", "")
        choices = example.get("choices", {}).get("text", [])
        answer = example.get("answerKey")
    else:
        question = example.get("question", "")
        choices = example.get("choices", {}).get("text", [])
        answer = example.get("answerKey")
    choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    prompt = build_prompt(dataset=dataset, locale="", question=question, choices=choices_str, use_cot=False, use_template=True)
    return prompt, answer


def _generate_message_llm(teacher_model, teacher_tokenizer, question, choices, budget_bytes):
    prompt = f"Provide a concise hint to solve the question.\n\nQuestion:\n{question}\n\nChoices:\n{choices}\n"
    messages = [{"role": "user", "content": prompt}]
    text = teacher_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = teacher_tokenizer(text, return_tensors="pt").to(teacher_model.device)
    with torch.no_grad():
        outputs = teacher_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    msg = teacher_tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Truncate to budget.
    msg_bytes = msg.encode("utf-8")
    if budget_bytes is not None and len(msg_bytes) > budget_bytes:
        msg = msg_bytes[:budget_bytes].decode("utf-8", errors="ignore")
    return msg


def eval_text_baseline(args, run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "results").mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.mode == "gpu" else "cpu")
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    set_default_chat_template(receiver_tokenizer, args.base_model)
    receiver = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()

    teacher = None
    teacher_tokenizer = None
    if args.text_style == "text_summary_llm":
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        set_default_chat_template(teacher_tokenizer, args.teacher_model)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()

    bytes_measured = {}
    bytes_breakdown = {}
    for dataset in args.eval_datasets.split(","):
        dataset = dataset.strip()
        data = _load_dataset(dataset)
        if args.eval_limit:
            data = data.select(range(int(args.eval_limit)))
        correct = 0
        total = 0
        bytes_list = []
        trunc_count = 0
        encode_times = []
        decode_times = []
        for idx, example in enumerate(data):
            prompt, answer = _format_example(example, dataset)
            question, choices = prompt.split("Choices:\n", 1)
            t_encode = time.perf_counter()
            message = make_message(
                example,
                budget_bytes=args.budget_bytes,
                style=args.text_style,
                llm_generate_fn=(
                    (lambda q, c, b: _generate_message_llm(teacher, teacher_tokenizer, q, c, b))
                    if teacher is not None else None
                ),
            )
            encode_times.append((time.perf_counter() - t_encode) * 1000.0)
            bytes_list.append(message["bytes"])
            if message.get("truncated"):
                trunc_count += 1
            t_decode = time.perf_counter()
            receiver_prompt = apply_payload_to_receiver(prompt, message, mode="text")
            decode_times.append((time.perf_counter() - t_decode) * 1000.0)
            inputs = receiver_tokenizer.apply_chat_template(
                [{"role": "user", "content": receiver_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = receiver_tokenizer(inputs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = receiver.generate(**model_inputs, do_sample=False, max_new_tokens=64)
            generated = outputs[0][model_inputs["input_ids"].shape[1]:]
            text = receiver_tokenizer.decode(generated, skip_special_tokens=True).strip()
            pred = extract_answer_from_content(text)
            if pred is not None and answer is not None and pred == answer:
                correct += 1
            total += 1
        accuracy = correct / total if total else None
        summary = {
            "dataset": dataset,
            "overall_accuracy": accuracy,
            "num_samples": total,
            "bytes_measured_total": sum(bytes_list) / len(bytes_list) if bytes_list else None,
            "truncation_rate": (trunc_count / total) if total else None,
            "encode_ms_mean": (sum(encode_times) / len(encode_times)) if encode_times else None,
            "decode_ms_mean": (sum(decode_times) / len(decode_times)) if decode_times else None,
        }
        slug = dataset.replace("/", "_")
        bytes_measured[slug] = summary["bytes_measured_total"]
        bytes_breakdown[slug] = {
            "payload_bytes": summary["bytes_measured_total"],
            "index_bytes": 0.0,
            "scale_bytes": 0.0,
            "header_bytes": 0.0,
        }
        out_dir = run_root / "results" / (dataset.replace("/", "_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"baseline_text_{dataset}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    manifest = {
        "baseline_family": "text",
        "bytes_estimate": bytes_measured,
        "bytes_estimated_total": bytes_measured,
        "bytes_measured_total": bytes_measured,
        "bytes_measured_breakdown": bytes_breakdown,
        "base_model": args.base_model,
        "teacher_model": args.teacher_model,
        "text_style": args.text_style,
        "budget_bytes": args.budget_bytes,
        "text_generation_source": "teacher" if args.text_style == "text_summary_llm" else "heuristic",
        "text_generation_settings": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 128,
        },
    }
    (run_root / "manifests" / "step_13_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run Step 13 baselines (text + kvcomm-like/qkvcomm-like).")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--mode", choices=["gpu", "local"], default="local")
    parser.add_argument("--baseline", choices=["text", "kvcomm_like", "qkvcomm_like"], default="text")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--eval-datasets", default="openbookqa,ai2-arc")
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--text-style", choices=["text_raw", "text_summary_heur", "text_summary_llm"], default="text_summary_heur")
    parser.add_argument("--budget-bytes", type=int, default=2048)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = project_root / "quantization" / "data" / "step_13_baselines"
    if args.run_tag is None:
        args.run_tag = time.strftime("step13_%Y%m%d_%H%M%S")
    run_root = data_root / args.run_tag

    if args.baseline == "text":
        eval_text_baseline(args, run_root)
        return

    # For kvcomm_like / qkvcomm_like, defer to step8 runner with appropriate flags.
    cmd = [
        sys.executable,
        str(project_root / "quantization" / "scripts" / "run_step8_selective_transfer.py"),
        "--mode", args.mode,
        "--run-tag", args.run_tag,
        "--base-model-override", args.base_model,
        "--teacher-model-override", args.teacher_model,
        "--wire-format", "kvwire_v1",
        "--wire-apply-pack",
        "--wire-record-per-sample",
        "--eval-datasets", args.eval_datasets,
    ]
    if args.baseline == "kvcomm_like":
        cmd += ["--kv-select-mode", "vnorm_topk", "--kv-select-proportion", "0.25"]
    else:
        # qkvcomm_like: use mixed precision schedule (last 2 layers int8).
        cmd += ["--kv-quant-scheme", "int4"]
    subprocess.check_call(cmd, cwd=str(project_root))
    manifest_path = project_root / "quantization" / "data" / "step_8_selective_transfer" / args.run_tag / "manifests" / "step_8_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
        manifest["baseline_family"] = args.baseline
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
