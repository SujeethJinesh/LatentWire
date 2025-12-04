#!/usr/bin/env python
# telepathy/eval_agnews_baselines.py
"""
AG News Baselines for Bridge Comparison

Baselines:
1. Random: 25% (trivial for 4-class)
2. Majority class: Based on label distribution
3. Mistral text: Mistral given full article text (upper bound)
4. Noise: Random soft tokens to Mistral (should be ~25%)
5. Llama text: Llama given full article text

These baselines tell us if bridge accuracy is meaningful.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

# AG News class labels
AGNEWS_LABELS = ["world", "sports", "business", "science"]

# Permissive matching for science/tech (AG News uses "Sci/Tech")
SCIENCE_SYNONYMS = ["science", "technology", "tech", "sci/tech", "scitech"]


def check_label_match(label, output):
    """Check if label matches output, with permissive matching for science."""
    if label == "science":
        return any(syn in output for syn in SCIENCE_SYNONYMS)
    return label in output


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
    class_correct = {l: 0 for l in AGNEWS_LABELS}
    class_total = {l: 0 for l in AGNEWS_LABELS}

    for i in tqdm(range(num_samples), desc=f"{model_name} text"):
        item = ds[i]
        text = item['text']
        label = AGNEWS_LABELS[item['label']]

        # Simple prompt
        prompt = f"Article: {text[:256]}\nTopic (world, sports, business, or science):"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Get only the generated part
            gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        # Check which label is in output (permissive for science/tech)
        is_correct = check_label_match(label, output)
        if is_correct:
            correct += 1
            class_correct[label] += 1
        total += 1
        class_total[label] += 1

    return 100 * correct / total, correct, total, class_correct, class_total


def eval_noise_baseline(model, tokenizer, ds, num_samples, device, tgt_dim):
    """Evaluate with random soft tokens (should be ~25%)."""
    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc="Noise baseline"):
        item = ds[i]
        label = AGNEWS_LABELS[item['label']]

        with torch.no_grad():
            # Random soft tokens (8 tokens to match optimal config, scaled)
            noise_tokens = torch.randn(1, 8, tgt_dim, device=device, dtype=torch.bfloat16) * 0.003

            # Primer
            primer = "Topic:"
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

        if check_label_match(label, output):
            correct += 1
        total += 1

    return 100 * correct / total, correct, total


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("AG NEWS BASELINES")
    print("=" * 60)
    print("These baselines tell us if bridge accuracy is meaningful.")
    print(f"Classes: {', '.join(AGNEWS_LABELS)}")
    print("")

    # Load AG News
    ds = load_dataset("ag_news", split="test")
    num_samples = min(args.num_samples, len(ds))

    # Class distribution
    labels = [item['label'] for item in ds]
    class_counts = {AGNEWS_LABELS[i]: labels.count(i) for i in range(4)}
    majority_class = max(class_counts, key=class_counts.get)
    majority_pct = 100 * class_counts[majority_class] / len(labels)

    print(f"Dataset: {len(ds)} samples")
    for label in AGNEWS_LABELS:
        pct = 100 * class_counts[label] / len(labels)
        print(f"  {label}: {class_counts[label]} ({pct:.1f}%)")
    print(f"  Majority class ({majority_class}): {majority_pct:.1f}%")
    print("")

    results = {
        "random_baseline": 25.0,
        "majority_baseline": majority_pct,
        "majority_class": majority_class,
        "class_distribution": class_counts
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
    mistral_acc, mistral_correct, mistral_total, m_class_correct, m_class_total = eval_text_baseline(
        mistral, mistral_tok, ds, num_samples, DEVICE, "Mistral"
    )
    print(f"  Mistral text accuracy: {mistral_correct}/{mistral_total} ({mistral_acc:.1f}%)")
    for label in AGNEWS_LABELS:
        if m_class_total[label] > 0:
            acc = 100 * m_class_correct[label] / m_class_total[label]
            print(f"    {label}: {m_class_correct[label]}/{m_class_total[label]} ({acc:.1f}%)")
    results["mistral_text_baseline"] = {
        "accuracy": mistral_acc,
        "correct": mistral_correct,
        "total": mistral_total,
        "per_class": {l: {"correct": m_class_correct[l], "total": m_class_total[l]} for l in AGNEWS_LABELS}
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
    llama_acc, llama_correct, llama_total, l_class_correct, l_class_total = eval_text_baseline(
        llama, llama_tok, ds, num_samples, DEVICE, "Llama"
    )
    print(f"  Llama text accuracy: {llama_correct}/{llama_total} ({llama_acc:.1f}%)")
    for label in AGNEWS_LABELS:
        if l_class_total[label] > 0:
            acc = 100 * l_class_correct[label] / l_class_total[label]
            print(f"    {label}: {l_class_correct[label]}/{l_class_total[label]} ({acc:.1f}%)")
    results["llama_text_baseline"] = {
        "accuracy": llama_acc,
        "correct": llama_correct,
        "total": llama_total,
        "per_class": {l: {"correct": l_class_correct[l], "total": l_class_total[l]} for l in AGNEWS_LABELS}
    }

    # Summary
    print("\n" + "=" * 60)
    print("AG NEWS BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Random:           25.0%")
    print(f"  Majority class:   {majority_pct:.1f}% ({majority_class})")
    print(f"  Noise (random):   {noise_acc:.1f}%")
    print(f"  Llama text:       {llama_acc:.1f}%")
    print(f"  Mistral text:     {mistral_acc:.1f}%  <- Upper bound")
    print("")

    # Interpretation
    if noise_acc > 35:
        print("WARNING: Noise baseline > 35% suggests Mistral has bias!")
    if mistral_acc < 70:
        print("NOTE: Mistral text baseline < 70% - task may be harder")
    else:
        print("Good: Mistral handles this task well.")
        print("Bridge target: Match or approach Mistral text baseline.")

    print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, "agnews_baselines.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
