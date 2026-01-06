#!/usr/bin/env python
# telepathy/eval_sst2_baselines.py
"""
SST-2 Baselines for Bridge Comparison

Baselines:
1. Random: 50% (trivial)
2. Majority class: Based on label distribution
3. Mistral text: Mistral given full review text (upper bound)
4. Noise: Random soft tokens to Mistral (should be ~50%)
5. Llama text: Llama given full review text

Without these, we can't know if 93.46% is impressive or an artifact.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def eval_text_baseline(model, tokenizer, ds, num_samples, device, model_name):
    """Evaluate model given full text (upper bound baseline)."""
    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc=f"{model_name} text"):
        item = ds[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        # Simple prompt
        prompt = f"Review: {text}\nSentiment (positive or negative):"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Get only the generated part
            gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        if label in output:
            correct += 1
        total += 1

    return 100 * correct / total, correct, total


def eval_noise_baseline(model, tokenizer, ds, num_samples, device, tgt_dim):
    """Evaluate with random soft tokens (should be ~50%)."""
    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc="Noise baseline"):
        item = ds[i]
        label = "positive" if item['label'] == 1 else "negative"

        with torch.no_grad():
            # Random soft tokens (32 tokens, scaled to typical embedding magnitude)
            noise_tokens = torch.randn(1, 32, tgt_dim, device=device, dtype=torch.bfloat16) * 0.003

            # Primer
            primer = "Sentiment:"
            primer_enc = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids)
            if primer_embeds.dtype != noise_tokens.dtype:
                primer_embeds = primer_embeds.to(noise_tokens.dtype)

            combined_embeds = torch.cat([primer_embeds, noise_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        if label in output:
            correct += 1
        total += 1

    return 100 * correct / total, correct, total


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SST-2 BASELINES")
    print("=" * 60)
    print("These baselines tell us if 93.46% bridge accuracy is real.")
    print("")

    # Load SST-2
    ds = load_dataset("glue", "sst2", split="validation")
    num_samples = min(args.num_samples, len(ds))

    # Class distribution
    labels = [item['label'] for item in ds]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    majority_pct = 100 * max(pos_count, neg_count) / len(labels)

    print(f"Dataset: {len(ds)} samples")
    print(f"  Positive: {pos_count} ({100*pos_count/len(labels):.1f}%)")
    print(f"  Negative: {neg_count} ({100*neg_count/len(labels):.1f}%)")
    print(f"  Majority class baseline: {majority_pct:.1f}%")
    print("")

    results = {
        "random_baseline": 50.0,
        "majority_baseline": majority_pct,
        "class_distribution": {
            "positive": pos_count,
            "negative": neg_count
        }
    }

    # Load Mistral
    print("Loading Mistral...")
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    # Mistral text baseline (upper bound)
    print("\n[1/3] Mistral Text Baseline (upper bound)...")
    mistral_acc, mistral_correct, mistral_total = eval_text_baseline(
        mistral, mistral_tok, ds, num_samples, DEVICE, "Mistral"
    )
    print(f"  Mistral text accuracy: {mistral_correct}/{mistral_total} ({mistral_acc:.1f}%)")
    results["mistral_text_baseline"] = {
        "accuracy": mistral_acc,
        "correct": mistral_correct,
        "total": mistral_total
    }

    # Noise baseline
    print("\n[2/3] Noise Baseline (random soft tokens)...")
    noise_acc, noise_correct, noise_total = eval_noise_baseline(
        mistral, mistral_tok, ds, num_samples, DEVICE, mistral.config.hidden_size
    )
    print(f"  Noise accuracy: {noise_correct}/{noise_total} ({noise_acc:.1f}%)")
    results["noise_baseline"] = {
        "accuracy": noise_acc,
        "correct": noise_correct,
        "total": noise_total
    }

    # Load Llama
    print("\nLoading Llama...")
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    # Llama text baseline
    print("\n[3/3] Llama Text Baseline...")
    llama_acc, llama_correct, llama_total = eval_text_baseline(
        llama, llama_tok, ds, num_samples, DEVICE, "Llama"
    )
    print(f"  Llama text accuracy: {llama_correct}/{llama_total} ({llama_acc:.1f}%)")
    results["llama_text_baseline"] = {
        "accuracy": llama_acc,
        "correct": llama_correct,
        "total": llama_total
    }

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Random:           50.0%")
    print(f"  Majority class:   {majority_pct:.1f}%")
    print(f"  Noise (random):   {noise_acc:.1f}%")
    print(f"  Llama text:       {llama_acc:.1f}%")
    print(f"  Mistral text:     {mistral_acc:.1f}%  <- Upper bound")
    print("")
    print(f"  Bridge result:    93.46%")
    print("")

    # Interpretation
    if noise_acc > 60:
        print("WARNING: Noise baseline > 60% suggests Mistral has bias!")
        print("  The bridge might just be triggering a default response.")
    elif mistral_acc < 85:
        print("NOTE: Mistral text baseline < 85%")
        print("  Bridge at 93.46% would exceed upper bound - suspicious!")
    else:
        gap = mistral_acc - 93.46
        print(f"GAP to upper bound: {gap:.1f}%")
        if gap < 5:
            print("  Bridge is near-optimal for this task!")
        else:
            print("  Bridge has room to improve.")

    print("=" * 60)

    # Save results
    import os
    output_path = os.path.join(args.output_dir, "sst2_baselines.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
