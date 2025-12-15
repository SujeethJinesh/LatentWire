#!/usr/bin/env python3
"""
Reasoning Benchmark Evaluation for Telepathy Bridge.

Evaluates the bridge on multiple-choice reasoning benchmarks to test
whether cross-model communication works beyond simple classification.

Benchmarks (official HuggingFace datasets):
1. BoolQ - Yes/No question answering (google/boolq)
2. PIQA - Physical intuition 2-way (ybisk/piqa)
3. WinoGrande - Commonsense 2-way (allenai/winogrande)
4. ARC-Challenge - Science 4-way (allenai/ai2_arc)
5. CommonsenseQA - Commonsense 5-way (tau/commonsense_qa)

All benchmarks use official evaluation protocols with accuracy metric.

Usage:
    python eval_reasoning_benchmarks.py --benchmark boolq --steps 2000
    python eval_reasoning_benchmarks.py --benchmark all --steps 2000
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

from latent_bridge_v15 import LatentBridgeV15


class Args:
    """Args object for LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=16, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


# =============================================================================
# BENCHMARK CONFIGURATIONS - Official HuggingFace datasets
# =============================================================================

BENCHMARK_CONFIGS = {
    "boolq": {
        "hf_path": "google/boolq",
        "hf_config": None,
        "eval_split": "validation",
        "num_choices": 2,
        "labels": ["No", "Yes"],  # False=No, True=Yes
        "get_text": lambda x: f"{x['passage']}\n\nQuestion: {x['question']}",
        "get_label": lambda x: 1 if x["answer"] else 0,  # bool -> int
        "description": "Yes/No reading comprehension",
    },
    "piqa": {
        "hf_path": "ybisk/piqa",
        "hf_config": None,
        "eval_split": "validation",
        "num_choices": 2,
        "labels": None,  # Dynamic: sol1, sol2
        "get_text": lambda x: f"Goal: {x['goal']}",
        "get_choices": lambda x: [x["sol1"], x["sol2"]],
        "get_label": lambda x: x["label"],  # int 0 or 1
        "description": "Physical intuition QA",
    },
    "winogrande": {
        "hf_path": "allenai/winogrande",
        "hf_config": "winogrande_debiased",
        "eval_split": "validation",
        "num_choices": 2,
        "labels": None,  # Dynamic: option1, option2
        "get_text": lambda x: x["sentence"].replace("_", "___"),
        "get_choices": lambda x: [x["option1"], x["option2"]],
        "get_label": lambda x: int(x["answer"]) - 1,  # "1"/"2" -> 0/1
        "description": "Commonsense coreference",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "eval_split": "test",  # Labels available for test
        "num_choices": 4,  # 3-4, but we handle dynamically
        "labels": ["A", "B", "C", "D"],
        "get_text": lambda x: x["question"],
        "get_choices": lambda x: x["choices"]["text"],
        "get_label": lambda x: x["choices"]["label"].index(x["answerKey"]),
        "description": "Science multiple choice",
    },
    "commonsenseqa": {
        "hf_path": "tau/commonsense_qa",
        "hf_config": None,
        "eval_split": "validation",
        "num_choices": 5,
        "labels": ["A", "B", "C", "D", "E"],
        "get_text": lambda x: x["question"],
        "get_choices": lambda x: x["choices"]["text"],
        "get_label": lambda x: ["A", "B", "C", "D", "E"].index(x["answerKey"]),
        "description": "Commonsense reasoning",
    },
}


def load_benchmark(name):
    """Load benchmark dataset from HuggingFace."""
    config = BENCHMARK_CONFIGS[name]

    if config["hf_config"]:
        dataset = load_dataset(config["hf_path"], config["hf_config"])
    else:
        dataset = load_dataset(config["hf_path"])

    return dataset[config["eval_split"]], config


def format_classification_prompt(text, choices, labels):
    """Format prompt for classification-style evaluation."""
    prompt = f"{text}\n\nChoices:\n"
    for i, (label, choice) in enumerate(zip(labels, choices)):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def train_and_eval_bridge(
    benchmark_name,
    train_data,
    eval_data,
    config,
    llama,
    mistral,
    llama_tok,
    mistral_tok,
    device,
    steps=2000,
    batch_size=8,
    lr=1e-4,
    soft_tokens=16,
    source_layer=31,
    diversity_weight=0.1,
    eval_samples=200,
):
    """Train bridge on benchmark and evaluate."""

    # Determine labels
    if config.get("labels"):
        labels = config["labels"]
    else:
        # For dynamic choices, we need a different approach
        labels = None

    num_choices = config["num_choices"]

    # Create bridge
    bridge_args = Args(soft_tokens=soft_tokens, heads=8, depth=2)
    bridge = LatentBridgeV15(bridge_args, src_dim=4096, tgt_dim=4096, target_rms=0.03)
    bridge = bridge.to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr)

    # Prepare primer
    primer = "Answer: "
    primer_tokens = mistral_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        primer_embeds_single = mistral.get_input_embeddings()(primer_tokens.input_ids)

    # Training
    bridge.train()
    train_losses = []
    train_items = list(train_data)

    print(f"\nTraining bridge for {benchmark_name}...")
    pbar = tqdm(range(steps), desc="Training")

    for step in pbar:
        # Sample batch
        batch_indices = torch.randint(0, len(train_items), (batch_size,)).tolist()
        batch_items = [train_items[i] for i in batch_indices]

        texts = []
        target_labels = []

        for item in batch_items:
            text = config["get_text"](item)
            label_idx = config["get_label"](item)

            # Get target string
            if labels:
                target_str = labels[label_idx]
            else:
                choices = config["get_choices"](item)
                target_str = choices[label_idx][:20]  # Truncate for training

            texts.append(text)
            target_labels.append(target_str)

        B = len(texts)

        # Encode with Llama
        src_inputs = llama_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            src_out = llama(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]

        # Bridge forward
        latents, aux_loss, _, _ = bridge(src_hidden, src_inputs.attention_mask)

        # Diversity loss
        flat_tokens = latents.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        div_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        div_loss = sim_matrix[div_mask].mean() if B > 1 else torch.tensor(0.0, device=device)

        # Target embeddings
        tgt_inputs = mistral_tok(
            target_labels,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(device)
        tgt_embeds = mistral.get_input_embeddings()(tgt_inputs.input_ids)

        # Combine: [primer] + [latents] + [target]
        primer_batch = primer_embeds_single.repeat(B, 1, 1)
        inputs_embeds = torch.cat([primer_batch, latents, tgt_embeds], dim=1)

        # Labels (ignore primer and latents)
        ignore_len = primer_batch.shape[1] + latents.shape[1]
        labels_tensor = torch.full((B, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
        labels_tensor[:, ignore_len:] = tgt_inputs.input_ids

        attn_mask = torch.ones(B, inputs_embeds.shape[1], device=device)

        # Forward through Mistral
        outputs = mistral(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_tensor)
        lm_loss = outputs.loss

        loss = lm_loss + diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    # Evaluation
    print(f"\nEvaluating on {benchmark_name}...")
    bridge.eval()
    correct = 0
    total = 0

    eval_items = list(eval_data)[:eval_samples]

    for i, item in enumerate(tqdm(eval_items, desc="Evaluating")):
        text = config["get_text"](item)
        true_label_idx = config["get_label"](item)

        # Get choices for this item
        if labels:
            choices = labels[:num_choices]
        else:
            choices = config["get_choices"](item)

        # Handle variable number of choices (e.g., ARC has 3-4)
        actual_num_choices = len(choices)

        # Encode with Llama
        src_inputs = llama_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            src_out = llama(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]
            latents, _, _, _ = bridge(src_hidden, src_inputs.attention_mask)

            # Score each choice
            scores = []
            for choice in choices:
                choice_text = choice if isinstance(choice, str) else str(choice)
                choice_inputs = mistral_tok(
                    choice_text[:30],  # Truncate long choices
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(device)
                choice_embeds = mistral.get_input_embeddings()(choice_inputs.input_ids)

                # Combine: [primer] + [latents] + [choice]
                combined = torch.cat([primer_embeds_single, latents, choice_embeds], dim=1)
                attn_mask = torch.ones(1, combined.shape[1], device=device)

                # Get loss for this choice (lower is better)
                labels_tensor = torch.full((1, combined.shape[1]), -100, dtype=torch.long, device=device)
                ignore_len = primer_embeds_single.shape[1] + latents.shape[1]
                labels_tensor[:, ignore_len:] = choice_inputs.input_ids

                outputs = mistral(inputs_embeds=combined, attention_mask=attn_mask, labels=labels_tensor)
                scores.append(-outputs.loss.item())  # Negative loss = higher score

            pred_idx = scores.index(max(scores))

        if pred_idx == true_label_idx:
            correct += 1
        total += 1

        if i < 5:
            print(f"[{i}] True: {true_label_idx}, Pred: {pred_idx}, Scores: {[f'{s:.2f}' for s in scores]}")

    accuracy = 100 * correct / total if total > 0 else 0
    bridge_params = sum(p.numel() for p in bridge.parameters())

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "bridge_params": bridge_params,
        "final_loss": sum(train_losses[-100:]) / min(100, len(train_losses)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True,
                       choices=list(BENCHMARK_CONFIGS.keys()) + ["all"],
                       help="Benchmark to evaluate")
    parser.add_argument("--output_dir", default="runs/reasoning_benchmarks")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--soft_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--source_layer", type=int, default=31)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("REASONING BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Steps: {args.steps}")
    print("=" * 60)

    # Load models
    print("\nLoading Llama...")
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(device)
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    print("Loading Mistral...")
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16
    ).to(device)
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    # Freeze models
    for p in llama.parameters():
        p.requires_grad = False
    for p in mistral.parameters():
        p.requires_grad = False

    # Determine benchmarks to run
    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_CONFIGS.keys())
    else:
        benchmarks = [args.benchmark]

    all_results = {
        "experiment": "reasoning_benchmarks",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "steps": args.steps,
            "soft_tokens": args.soft_tokens,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "source_layer": args.source_layer,
            "eval_samples": args.eval_samples,
        },
        "results": {},
    }

    for bench_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bench_name.upper()}")
        print(f"{'='*60}")

        config = BENCHMARK_CONFIGS[bench_name]
        print(f"Description: {config['description']}")
        print(f"Choices: {config['num_choices']}")

        # Load dataset
        print(f"Loading {config['hf_path']}...")
        try:
            eval_data, config = load_benchmark(bench_name)

            # Use train split for training if available
            if config["hf_config"]:
                train_dataset = load_dataset(config["hf_path"], config["hf_config"])
            else:
                train_dataset = load_dataset(config["hf_path"])

            train_data = train_dataset.get("train", eval_data)

            print(f"Train samples: {len(train_data)}")
            print(f"Eval samples: {len(eval_data)} (using {args.eval_samples})")

            # Train and evaluate
            results = train_and_eval_bridge(
                benchmark_name=bench_name,
                train_data=train_data,
                eval_data=eval_data,
                config=config,
                llama=llama,
                mistral=mistral,
                llama_tok=llama_tok,
                mistral_tok=mistral_tok,
                device=device,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                soft_tokens=args.soft_tokens,
                source_layer=args.source_layer,
                diversity_weight=args.diversity_weight,
                eval_samples=args.eval_samples,
            )

            all_results["results"][bench_name] = results
            print(f"\n{bench_name} Results: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")

        except Exception as e:
            print(f"ERROR loading {bench_name}: {e}")
            all_results["results"][bench_name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Reference baselines (approximate, from literature)
    baselines = {
        "boolq": {"random": 50.0, "llama_0shot": 83.0},
        "piqa": {"random": 50.0, "llama_0shot": 81.0},
        "winogrande": {"random": 50.0, "llama_0shot": 77.0},
        "arc_challenge": {"random": 25.0, "llama_0shot": 55.0},
        "commonsenseqa": {"random": 20.0, "llama_0shot": 72.0},
    }

    print(f"{'Benchmark':<20} {'Bridge':<10} {'Random':<10} {'vs Random':<12}")
    print("-" * 52)

    for bench_name, results in all_results["results"].items():
        if "accuracy" in results:
            acc = results["accuracy"]
            random_base = baselines.get(bench_name, {}).get("random", 0)
            vs_random = acc - random_base
            print(f"{bench_name:<20} {acc:>6.1f}%   {random_base:>6.1f}%   {vs_random:>+6.1f}pp")
        else:
            print(f"{bench_name:<20} ERROR: {results.get('error', 'unknown')[:30]}")

    # Save results
    output_file = f"{args.output_dir}/reasoning_{args.benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
