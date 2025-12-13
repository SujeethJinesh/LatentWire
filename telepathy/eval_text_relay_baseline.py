#!/usr/bin/env python
# telepathy/eval_text_relay_baseline.py
"""
Text-Relay Baseline: Llama summarizes → text → Mistral classifies

This baseline answers: "Is the bridge benefit from:
  (a) Llama's encoding being better than Mistral's reading? OR
  (b) Something special about the latent space?"

If text-relay matches bridge performance, the benefit is from (a).
If bridge >> text-relay, there's something special about latent transfer.
"""
import os
import torch
import json
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def generate_summary(model, tokenizer, text, max_input_len=512, max_summary_len=50):
    """Have Llama generate a short summary of the input."""
    prompt = f"Summarize this in one sentence:\n\n{text[:max_input_len]}\n\nSummary:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len + 50)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_summary_len,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    summary = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return summary.strip()


def classify_with_mistral(model, tokenizer, summary, task_prompt, device):
    """Have Mistral classify based on Llama's summary."""
    prompt = f"{task_prompt}\n\nText: {summary}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip().lower()


def eval_sst2(llama_model, llama_tok, mistral_model, mistral_tok, device, num_samples=200):
    """Evaluate text-relay on SST-2."""
    print("\n" + "=" * 60)
    print("SST-2 TEXT-RELAY BASELINE")
    print("=" * 60)

    dataset = load_dataset("glue", "sst2", split="validation")

    task_prompt = "Is the sentiment positive or negative?"

    correct = 0
    total = 0

    for i in tqdm(range(min(num_samples, len(dataset))), desc="SST-2"):
        item = dataset[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        # Step 1: Llama summarizes
        summary = generate_summary(llama_model, llama_tok, text)

        # Step 2: Mistral classifies from summary
        prediction = classify_with_mistral(mistral_model, mistral_tok, summary, task_prompt, device)

        if label in prediction:
            correct += 1
        total += 1

        if i < 3:
            print(f"\n[{i}] Original: {text[:60]}...")
            print(f"    Summary: {summary[:60]}...")
            print(f"    Label: {label}, Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nSST-2 Text-Relay Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return {"accuracy": accuracy, "correct": correct, "total": total}


def eval_agnews(llama_model, llama_tok, mistral_model, mistral_tok, device, num_samples=200):
    """Evaluate text-relay on AG News."""
    print("\n" + "=" * 60)
    print("AG NEWS TEXT-RELAY BASELINE")
    print("=" * 60)

    dataset = load_dataset("ag_news", split="test")
    labels = ["world", "sports", "business", "science"]

    task_prompt = "What is the topic: world, sports, business, or science?"

    correct = 0
    total = 0

    # Permissive matching for science
    science_synonyms = ["science", "technology", "tech", "sci/tech", "scitech"]

    for i in tqdm(range(min(num_samples, len(dataset))), desc="AG News"):
        item = dataset[i]
        text = item['text']
        label = labels[item['label']]

        # Step 1: Llama summarizes
        summary = generate_summary(llama_model, llama_tok, text[:500])

        # Step 2: Mistral classifies from summary
        prediction = classify_with_mistral(mistral_model, mistral_tok, summary, task_prompt, device)

        # Check match (permissive for science)
        if label == "science":
            is_correct = any(syn in prediction for syn in science_synonyms)
        else:
            is_correct = label in prediction

        if is_correct:
            correct += 1
        total += 1

        if i < 3:
            print(f"\n[{i}] Original: {text[:60]}...")
            print(f"    Summary: {summary[:60]}...")
            print(f"    Label: {label}, Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nAG News Text-Relay Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/text_relay_baseline")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--llama_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--mistral_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TEXT-RELAY BASELINE")
    print("=" * 60)
    print("Pipeline: Llama summarizes → text → Mistral classifies")
    print(f"Llama: {args.llama_model}")
    print(f"Mistral: {args.mistral_model}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Llama
    print("\nLoading Llama...")
    llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
    llama_tok.pad_token = llama_tok.eos_token
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.llama_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    llama_model.eval()

    # Load Mistral
    print("Loading Mistral...")
    mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
    mistral_tok.pad_token = mistral_tok.eos_token
    mistral_model = AutoModelForCausalLM.from_pretrained(
        args.mistral_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    mistral_model.eval()

    results = {
        "experiment": "text_relay_baseline",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llama_model": args.llama_model,
            "mistral_model": args.mistral_model,
            "num_samples": args.num_samples,
        },
        "results": {}
    }

    # Run evaluations
    results["results"]["sst2"] = eval_sst2(
        llama_model, llama_tok, mistral_model, mistral_tok, device, args.num_samples
    )

    results["results"]["agnews"] = eval_agnews(
        llama_model, llama_tok, mistral_model, mistral_tok, device, args.num_samples
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEXT-RELAY BASELINE SUMMARY")
    print("=" * 60)
    print(f"SST-2:   {results['results']['sst2']['accuracy']:.1f}%")
    print(f"AG News: {results['results']['agnews']['accuracy']:.1f}%")
    print("=" * 60)

    # Compare with bridge results
    print("\nComparison with Bridge:")
    print("| Task    | Text-Relay | Bridge | Mistral Text |")
    print("|---------|------------|--------|--------------|")
    print(f"| SST-2   | {results['results']['sst2']['accuracy']:.1f}%      | 94.7%  | 93.5%        |")
    print(f"| AG News | {results['results']['agnews']['accuracy']:.1f}%      | 88.9%  | 70.5%        |")

    # Save results
    json_path = os.path.join(args.output_dir, "text_relay_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
