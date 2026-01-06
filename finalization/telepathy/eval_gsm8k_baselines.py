#!/usr/bin/env python
# telepathy/eval_gsm8k_baselines.py
"""
GSM8K Baselines for Bridge Comparison

Baselines:
1. Random: ~0% (guessing numbers is essentially impossible)
2. Mistral text: Mistral given full question with 8-shot CoT (upper bound)
3. Llama text: Llama given full question with 8-shot CoT
4. Noise: Random soft tokens to Mistral (should be ~0%)

Uses standard lm-evaluation-harness format:
- 8-shot chain-of-thought prompting
- "Q: {question}\n A:" format
- "The answer is X" extraction

These baselines tell us if bridge accuracy is meaningful.
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os
import random


# Standard 8-shot CoT examples from lm-evaluation-harness
# These follow the exact format used in the official evaluation
FEW_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.

"""


def extract_answer(text):
    """Extract numeric answer following standard GSM8K format."""
    # Primary: Look for "The answer is X" (standard CoT format)
    match = re.search(r'[Tt]he answer is\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')

    # Secondary: Look for #### pattern (dataset format)
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
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def eval_text_baseline(model, tokenizer, ds, num_samples, device, model_name):
    """Evaluate model with 8-shot CoT prompting (standard GSM8K format)."""
    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc=f"{model_name} 8-shot CoT"):
        item = ds[i]
        question = item['question']
        gold_answer = extract_answer(item['answer'])

        if gold_answer is None:
            continue

        # Standard 8-shot CoT format from lm-evaluation-harness
        prompt = FEW_SHOT_EXAMPLES + f"Q: {question}\nA:"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=256,  # Need tokens for CoT reasoning
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Get only the generated part
            gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Extract predicted answer
        pred_answer = extract_answer(output)
        if pred_answer is None:
            numbers = re.findall(r'-?\d+', output)
            pred_answer = numbers[-1] if numbers else "?"

        is_correct = str(pred_answer) == str(gold_answer)
        if is_correct:
            correct += 1
        total += 1

        # Print first 3 examples
        if i < 3:
            status = "CORRECT" if is_correct else "wrong"
            print(f"\n  [{i}] {status}")
            print(f"    Q: {question[:50]}...")
            print(f"    Gold: {gold_answer} | Pred: {pred_answer}")
            print(f"    Output: {output[:100]}...")

    return 100 * correct / max(total, 1), correct, total


def eval_noise_baseline(model, tokenizer, ds, num_samples, device, tgt_dim, num_tokens=32):
    """Evaluate with random soft tokens (should be ~0%)."""
    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc="Noise baseline"):
        item = ds[i]
        gold_answer = extract_answer(item['answer'])

        if gold_answer is None:
            continue

        with torch.no_grad():
            # Random soft tokens (scaled to match typical embedding RMS)
            noise_tokens = torch.randn(1, num_tokens, tgt_dim, device=device, dtype=torch.bfloat16) * 0.003

            # Primer
            primer = "The answer is"
            primer_enc = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids)
            if primer_embeds.dtype != noise_tokens.dtype:
                primer_embeds = primer_embeds.to(noise_tokens.dtype)

            combined_embeds = torch.cat([primer_embeds, noise_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # Extract predicted answer
        pred_answer = extract_answer(output)
        if pred_answer is None:
            numbers = re.findall(r'-?\d+', output)
            pred_answer = numbers[0] if numbers else "?"

        is_correct = str(pred_answer) == str(gold_answer)
        if is_correct:
            correct += 1
        total += 1

    return 100 * correct / max(total, 1), correct, total


def eval_random_baseline(ds, num_samples):
    """Evaluate random number guessing (essentially 0%)."""
    correct = 0
    total = 0

    # Collect answer distribution to guess from
    answers = []
    for i in range(min(1000, len(ds))):
        ans = extract_answer(ds[i]['answer'])
        if ans:
            answers.append(ans)

    for i in range(num_samples):
        item = ds[i]
        gold_answer = extract_answer(item['answer'])

        if gold_answer is None:
            continue

        # Random guess from answer distribution
        pred_answer = random.choice(answers) if answers else "0"

        is_correct = str(pred_answer) == str(gold_answer)
        if is_correct:
            correct += 1
        total += 1

    return 100 * correct / max(total, 1), correct, total


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("GSM8K BASELINES")
    print("=" * 60)
    print("These baselines tell us if bridge accuracy is meaningful.")
    print("")

    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="test")
    num_samples = min(args.num_samples, len(ds))

    # Answer distribution analysis
    answers = []
    for i in range(len(ds)):
        ans = extract_answer(ds[i]['answer'])
        if ans:
            try:
                answers.append(float(ans))
            except:
                pass

    print(f"Dataset: {len(ds)} test samples")
    print(f"Answer range: {min(answers):.0f} to {max(answers):.0f}")
    print(f"Median answer: {sorted(answers)[len(answers)//2]:.0f}")
    print("")

    results = {}

    # Random baseline
    print("[1/4] Random Baseline (guessing from answer distribution)...")
    random_acc, random_correct, random_total = eval_random_baseline(ds, num_samples)
    print(f"  Random accuracy: {random_correct}/{random_total} ({random_acc:.1f}%)")
    results["random_baseline"] = {
        "accuracy": random_acc,
        "correct": random_correct,
        "total": random_total
    }

    # Load Mistral
    print("\nLoading Mistral...")
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    # Noise baseline
    print("\n[2/4] Noise Baseline (random soft tokens)...")
    noise_acc, noise_correct, noise_total = eval_noise_baseline(
        mistral, mistral_tok, ds, num_samples, DEVICE, mistral.config.hidden_size
    )
    print(f"  Noise accuracy: {noise_correct}/{noise_total} ({noise_acc:.1f}%)")
    results["noise_baseline"] = {
        "accuracy": noise_acc,
        "correct": noise_correct,
        "total": noise_total
    }

    # Mistral text baseline (upper bound)
    print("\n[3/4] Mistral Text Baseline (upper bound)...")
    mistral_acc, mistral_correct, mistral_total = eval_text_baseline(
        mistral, mistral_tok, ds, num_samples, DEVICE, "Mistral"
    )
    print(f"  Mistral text accuracy: {mistral_correct}/{mistral_total} ({mistral_acc:.1f}%)")
    results["mistral_text_baseline"] = {
        "accuracy": mistral_acc,
        "correct": mistral_correct,
        "total": mistral_total
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
    print("\n[4/4] Llama Text Baseline...")
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
    print("GSM8K BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Random:           {random_acc:.1f}%")
    print(f"  Noise (random):   {noise_acc:.1f}%")
    print(f"  Llama text:       {llama_acc:.1f}%")
    print(f"  Mistral text:     {mistral_acc:.1f}%  <- Upper bound")
    print("")

    # Interpretation
    print("For bridge to show meaningful reasoning transfer:")
    print(f"  - Must exceed noise baseline ({noise_acc:.1f}%)")
    print(f"  - Target: approach Mistral text ({mistral_acc:.1f}%)")
    print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, "gsm8k_baselines.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
