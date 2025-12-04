#!/usr/bin/env python
# telepathy/eval_telepathy_gsm8k.py
"""
Phase 18: GSM8K Evaluation with Latent Chain-of-Thought

Evaluates the trained CoT bridge on GSM8K math reasoning.

Success criteria:
  - Random baseline: ~0% (math is hard to guess)
  - Accuracy > 5%: Some signal is getting through
  - Accuracy > 15%: Bridge is working for reasoning
  - Accuracy > 30%: Excellent reasoning transfer
  - Accuracy matches Mistral text: Perfect transfer
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

from latent_cot_bridge import LatentCoTBridge


def extract_answer(text):
    """Extract numeric answer from GSM8K format '#### number'."""
    # Look for #### pattern
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')

    # Fallback: look for last number in text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=31)
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--cot_steps", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Phase 18 Evaluation: GSM8K Math Reasoning")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CoT Steps: {args.cot_steps}")
    print(f"Soft Tokens per Step: {args.soft_tokens}")
    print(f"Total Latent Tokens: {args.cot_steps * args.soft_tokens}")
    print("")
    print("Success Criteria:")
    print("  - Random baseline: ~0%")
    print("  - Accuracy > 5%: Some signal")
    print("  - Accuracy > 15%: Bridge works for reasoning")
    print("  - Accuracy > 30%: Excellent transfer")
    print("=" * 60)
    print("")

    # Load models
    print("Loading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Load bridge
    bridge = LatentCoTBridge(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    bridge.load_state_dict(checkpoint)
    bridge.to(DEVICE)
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load GSM8K test set
    ds = load_dataset("openai/gsm8k", "main", split="test")
    num_samples = min(args.num_samples, len(ds))

    print(f"\nEvaluating {num_samples} samples...")
    print("-" * 60)

    correct = 0
    total = 0
    results = []

    # Track answer magnitude distribution
    answer_magnitudes = {"small": 0, "medium": 0, "large": 0}
    correct_by_magnitude = {"small": 0, "medium": 0, "large": 0}

    for i in tqdm(range(num_samples), desc="Evaluating"):
        item = ds[i]
        question = item['question']
        gold_answer = extract_answer(item['answer'])

        if gold_answer is None:
            continue

        # Categorize answer magnitude
        try:
            ans_val = abs(float(gold_answer))
            if ans_val < 10:
                magnitude = "small"
            elif ans_val < 100:
                magnitude = "medium"
            else:
                magnitude = "large"
        except:
            magnitude = "medium"
        answer_magnitudes[magnitude] += 1

        # Source: Llama reads question
        src_input = f"Question: {question}\nLet me solve this step by step."
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()

            # Bridge with CoT
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Target: Mistral generates answer
            primer = "The answer is"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(DEVICE)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=DEVICE, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip()

        # Extract predicted answer
        pred_answer = extract_answer(output)
        if pred_answer is None:
            numbers = re.findall(r'-?\d+', output)
            pred_answer = numbers[0] if numbers else "?"

        is_correct = str(pred_answer) == str(gold_answer)
        if is_correct:
            correct += 1
            correct_by_magnitude[magnitude] += 1
        total += 1

        results.append({
            "idx": i,
            "question": question[:80] + "..." if len(question) > 80 else question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "output": output[:50],
            "correct": is_correct,
            "magnitude": magnitude
        })

        # Print first 10 samples
        if i < 10:
            status = "CORRECT" if is_correct else "wrong"
            print(f"\n[{i}] {status}")
            print(f"  Q: {question[:60]}...")
            print(f"  Gold: {gold_answer} | Pred: {pred_answer}")
            print(f"  Output: {output[:50]}")

    # Summary
    accuracy = 100 * correct / max(total, 1)

    print("\n" + "=" * 60)
    print("PHASE 18 GSM8K EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nOVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")

    print(f"\nBy answer magnitude:")
    for mag in ["small", "medium", "large"]:
        if answer_magnitudes[mag] > 0:
            mag_acc = 100 * correct_by_magnitude[mag] / answer_magnitudes[mag]
            print(f"  {mag:8}: {correct_by_magnitude[mag]:4}/{answer_magnitudes[mag]:4} ({mag_acc:.1f}%)")

    # Interpretation
    print("")
    if accuracy < 2:
        print("=" * 60)
        print("BASELINE: Near-random performance")
        print("Bridge is NOT transmitting reasoning signal yet.")
        print("This is expected early in training - need more steps.")
        print("=" * 60)
    elif accuracy < 10:
        print("=" * 60)
        print("SIGNAL DETECTED: Some reasoning transfer!")
        print("The bridge is learning to transmit math reasoning.")
        print("Continue training for better results.")
        print("=" * 60)
    elif accuracy < 25:
        print("=" * 60)
        print("SUCCESS: Bridge works for reasoning!")
        print("Good transfer of mathematical problem-solving.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXCELLENT: Strong reasoning transfer!")
        print("The latent CoT is effectively compressing chain-of-thought.")
        print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, "eval_gsm8k_results.json")
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "cot_steps": args.cot_steps,
        "soft_tokens_per_step": args.soft_tokens,
        "total_latent_tokens": args.cot_steps * args.soft_tokens,
        "by_magnitude": {
            mag: {
                "correct": correct_by_magnitude[mag],
                "total": answer_magnitudes[mag],
                "accuracy": 100 * correct_by_magnitude[mag] / max(answer_magnitudes[mag], 1)
            }
            for mag in ["small", "medium", "large"]
        },
        "samples": results[:50]
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
