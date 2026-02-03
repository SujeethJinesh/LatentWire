#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from rosetta.utils.evaluate import build_prompt, extract_answer_from_content, set_default_chat_template

from quantization.baselines.text_comm import make_message


def _load_dataset(name, limit=None):
    if name == "openbookqa":
        data = load_dataset("allenai/openbookqa", split="test")
    else:
        data = load_dataset("allenai/ai2_arc", split="test")
    if limit:
        data = data.select(range(int(limit)))
    return data


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


def _generate_answer(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _generate_message(model, tokenizer, question, choices, budget_bytes):
    prompt = f"Provide a concise hint.\n\nQuestion:\n{question}\n\nChoices:\n{choices}\n"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    msg = tokenizer.decode(generated, skip_special_tokens=True).strip()
    data = msg.encode("utf-8")
    if budget_bytes is not None and len(data) > budget_bytes:
        msg = data[:budget_bytes].decode("utf-8", errors="ignore")
    return msg


def main():
    parser = argparse.ArgumentParser(description="Run M15 two-round refinement pilot.")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--dataset", choices=["openbookqa", "ai2-arc"], default="openbookqa")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budget-round0", type=int, default=1024)
    parser.add_argument("--budget-round1", type=int, default=2048)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    run_tag = args.run_tag or time.strftime("step15_%Y%m%d_%H%M%S")
    run_root = project_root / "quantization" / "data" / "step_15_refine_pilot" / run_tag
    (run_root / "results").mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    set_default_chat_template(receiver_tokenizer, args.base_model)
    receiver = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    set_default_chat_template(teacher_tokenizer, args.teacher_model)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()

    data = _load_dataset(args.dataset, limit=args.limit)
    correct = 0
    total = 0
    round2_used = 0
    bytes_round0 = 0
    bytes_round1 = 0

    for example in data:
        prompt, answer = _format_example(example, args.dataset)
        msg0 = make_message(example, budget_bytes=args.budget_round0, style="text_summary_llm",
                            llm_generate_fn=lambda q, c, b: _generate_message(teacher, teacher_tokenizer, q, c, b))
        bytes_round0 += msg0["bytes"]
        prompt0 = f"Teacher message:\n{msg0['message']}\n\n{prompt}"
        text0 = _generate_answer(receiver, receiver_tokenizer, prompt0)
        pred0 = extract_answer_from_content(text0)
        use_round2 = pred0 is None
        if use_round2:
            round2_used += 1
            msg1 = make_message(example, budget_bytes=args.budget_round1, style="text_summary_llm",
                                llm_generate_fn=lambda q, c, b: _generate_message(teacher, teacher_tokenizer, q, c, b))
            bytes_round1 += msg1["bytes"]
            prompt1 = f"Teacher message:\n{msg1['message']}\n\n{prompt}"
            text1 = _generate_answer(receiver, receiver_tokenizer, prompt1)
            pred = extract_answer_from_content(text1)
        else:
            pred = pred0
        if pred is not None and answer is not None and pred == answer:
            correct += 1
        total += 1

    summary = {
        "dataset": args.dataset,
        "num_samples": total,
        "accuracy": correct / total if total else None,
        "round2_used": round2_used,
        "avg_bytes_round0": bytes_round0 / total if total else None,
        "avg_bytes_round1": bytes_round1 / round2_used if round2_used else None,
    }
    (run_root / "results" / f"pilot_{args.dataset}_summary.json").write_text(json.dumps(summary, indent=2))
    (run_root / "manifests" / "step_15_manifest.json").write_text(json.dumps({"baseline_family": "refine_pilot"}, indent=2))


if __name__ == "__main__":
    main()
