#!/usr/bin/env python3
"""
Bridge-Based Reviewer Response Experiments.

These experiments require training the actual bridge model and address:
- Colin Raffel: Same-checkpoint baseline, task interpolation
- Chelsea Finn: Minimum data experiments
- Denny Zhou: Self-consistency with bridge
- Sara Hooker: Soft token quantization

Usage:
    python run_bridge_reviewer_experiments.py --experiment all --output_dir runs/bridge_reviewer
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import bridge components (assuming they exist in the codebase)
try:
    from train_bridge import PerceiverResampler, train_bridge
    from eval_bridge import evaluate_bridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("Warning: Bridge training modules not found. Some experiments will be skipped.")


# =============================================================================
# EXPERIMENT: SAME-CHECKPOINT BASELINE (Colin Raffel)
# =============================================================================

def exp_same_checkpoint_baseline(output_dir, max_train_samples=2000, eval_samples=200):
    """
    Test if same-checkpoint (not just same-architecture) baseline differs.
    Addresses: Colin Raffel Q2 - "What happens if sender and receiver are same checkpoint?"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Same-Checkpoint Baseline")
    print("="*70)

    results = {
        "experiment": "same_checkpoint_baseline",
        "timestamp": datetime.now().isoformat(),
        "description": "Compare same-architecture vs identical-checkpoint bridges"
    }

    if not BRIDGE_AVAILABLE:
        results["status"] = "skipped"
        results["reason"] = "Bridge training modules not available"
        return results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    configs = [
        {
            "name": "same_checkpoint",
            "sender_id": model_id,
            "receiver_id": model_id,
            "share_model": True,  # Use same model instance
            "description": "Identical model instance for sender and receiver"
        },
        {
            "name": "same_architecture_diff_load",
            "sender_id": model_id,
            "receiver_id": model_id,
            "share_model": False,  # Load separately (may have different random states)
            "description": "Same architecture but separate model loads"
        },
    ]

    # Load dataset
    dataset = load_dataset("glue", "sst2", trust_remote_code=True)
    train_data = list(dataset["train"])[:max_train_samples]
    eval_data = list(dataset["validation"])[:eval_samples]

    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        print(f"  {config['description']}")

        try:
            # Train bridge
            bridge_results = train_bridge(
                sender_model_id=config["sender_id"],
                receiver_model_id=config["receiver_id"],
                train_data=train_data,
                eval_data=eval_data,
                share_model=config.get("share_model", False),
                num_soft_tokens=8,
                training_steps=2000,
                device=device,
            )

            results[config["name"]] = {
                "accuracy": bridge_results["accuracy"],
                "config": config,
                "training_loss": bridge_results.get("final_loss", None),
            }

            print(f"  Accuracy: {bridge_results['accuracy']:.1f}%")

        except Exception as e:
            results[config["name"]] = {"error": str(e)}
            print(f"  Error: {e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "same_checkpoint_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# EXPERIMENT: MINIMUM DATA (Chelsea Finn)
# =============================================================================

def exp_minimum_data(output_dir, eval_samples=200):
    """
    Find minimum training data needed for effective bridge.
    Addresses: Chelsea Finn Q3 - "What's the minimum training data needed?"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Minimum Data Requirements")
    print("="*70)

    results = {
        "experiment": "minimum_data",
        "timestamp": datetime.now().isoformat(),
        "description": "Training data ablation study"
    }

    if not BRIDGE_AVAILABLE:
        results["status"] = "skipped"
        results["reason"] = "Bridge training modules not available"
        return results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset
    dataset = load_dataset("glue", "sst2", trust_remote_code=True)
    full_train_data = list(dataset["train"])
    eval_data = list(dataset["validation"])[:eval_samples]

    # Test different training set sizes
    train_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

    for train_size in train_sizes:
        print(f"\n--- Training with {train_size} samples ---")

        # Sample training data
        np.random.seed(42)
        indices = np.random.choice(len(full_train_data), min(train_size, len(full_train_data)), replace=False)
        train_subset = [full_train_data[i] for i in indices]

        try:
            bridge_results = train_bridge(
                sender_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                receiver_model_id="mistralai/Mistral-7B-Instruct-v0.3",
                train_data=train_subset,
                eval_data=eval_data,
                num_soft_tokens=8,
                training_steps=min(2000, train_size * 4),  # Scale steps with data
                device=device,
            )

            results[f"train_{train_size}"] = {
                "train_samples": train_size,
                "accuracy": bridge_results["accuracy"],
                "training_steps": min(2000, train_size * 4),
                "final_loss": bridge_results.get("final_loss", None),
            }

            print(f"  Accuracy: {bridge_results['accuracy']:.1f}%")

        except Exception as e:
            results[f"train_{train_size}"] = {"error": str(e), "train_samples": train_size}
            print(f"  Error: {e}")

    # Compute data efficiency metrics
    if any("accuracy" in results.get(f"train_{s}", {}) for s in train_sizes):
        accuracies = [(s, results.get(f"train_{s}", {}).get("accuracy", 0)) for s in train_sizes]
        results["data_efficiency_summary"] = {
            "samples_vs_accuracy": accuracies,
            "min_samples_for_80pct": next((s for s, a in accuracies if a >= 80), None),
            "min_samples_for_90pct": next((s for s, a in accuracies if a >= 90), None),
        }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "minimum_data_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# EXPERIMENT: SELF-CONSISTENCY (Denny Zhou)
# =============================================================================

def exp_self_consistency(output_dir, num_runs=5, eval_samples=200):
    """
    Test self-consistency (multiple bridge runs with voting).
    Addresses: Denny Zhou - "Self-consistency / majority voting with bridge"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Self-Consistency Analysis")
    print("="*70)

    results = {
        "experiment": "self_consistency",
        "timestamp": datetime.now().isoformat(),
        "description": "Multiple bridge training runs with voting"
    }

    if not BRIDGE_AVAILABLE:
        results["status"] = "skipped"
        results["reason"] = "Bridge training modules not available"
        return results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset("glue", "sst2", trust_remote_code=True)
    train_data = list(dataset["train"])[:2000]
    eval_data = list(dataset["validation"])[:eval_samples]

    # Train multiple bridges with different seeds
    all_predictions = []
    individual_accuracies = []

    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_idx * 100 + 42}) ---")

        try:
            bridge_results = train_bridge(
                sender_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                receiver_model_id="mistralai/Mistral-7B-Instruct-v0.3",
                train_data=train_data,
                eval_data=eval_data,
                num_soft_tokens=8,
                training_steps=2000,
                seed=run_idx * 100 + 42,
                device=device,
                return_predictions=True,
            )

            all_predictions.append(bridge_results.get("predictions", []))
            individual_accuracies.append(bridge_results["accuracy"])

            print(f"  Accuracy: {bridge_results['accuracy']:.1f}%")

        except Exception as e:
            print(f"  Error: {e}")

    # Compute majority voting accuracy
    if len(all_predictions) >= 3:
        from collections import Counter

        majority_correct = 0
        for i in range(len(eval_data)):
            votes = [preds[i] if i < len(preds) else None for preds in all_predictions]
            votes = [v for v in votes if v is not None]
            if votes:
                majority_vote = Counter(votes).most_common(1)[0][0]
                true_label = eval_data[i].get("label", eval_data[i].get("label_field", 0))
                if majority_vote == true_label:
                    majority_correct += 1

        majority_accuracy = 100 * majority_correct / len(eval_data)
    else:
        majority_accuracy = None

    results["individual_accuracies"] = individual_accuracies
    results["mean_individual"] = np.mean(individual_accuracies) if individual_accuracies else None
    results["std_individual"] = np.std(individual_accuracies) if individual_accuracies else None
    results["majority_voting_accuracy"] = majority_accuracy
    results["num_runs"] = len(individual_accuracies)

    print(f"\n--- Summary ---")
    print(f"Individual mean: {results['mean_individual']:.1f}% ± {results['std_individual']:.1f}%")
    print(f"Majority voting: {results['majority_voting_accuracy']:.1f}%" if majority_accuracy else "N/A")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "self_consistency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# EXPERIMENT: SOFT TOKEN QUANTIZATION (Sara Hooker)
# =============================================================================

def exp_soft_token_quantization(output_dir, eval_samples=200):
    """
    Test effect of quantizing soft tokens to different precision.
    Addresses: Sara Hooker - "Can soft tokens be quantized without accuracy loss?"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Soft Token Quantization")
    print("="*70)

    results = {
        "experiment": "soft_token_quantization",
        "timestamp": datetime.now().isoformat(),
        "description": "Quantization ablation for soft tokens"
    }

    if not BRIDGE_AVAILABLE:
        results["status"] = "skipped"
        results["reason"] = "Bridge training modules not available"
        return results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset("glue", "sst2", trust_remote_code=True)
    train_data = list(dataset["train"])[:2000]
    eval_data = list(dataset["validation"])[:eval_samples]

    # Train bridge first
    print("\nTraining baseline bridge (fp32)...")
    bridge_results = train_bridge(
        sender_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        receiver_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        train_data=train_data,
        eval_data=eval_data,
        num_soft_tokens=8,
        training_steps=2000,
        device=device,
        return_soft_tokens=True,
    )

    baseline_accuracy = bridge_results["accuracy"]
    soft_tokens = bridge_results.get("soft_tokens", None)

    results["baseline_fp32"] = {"accuracy": baseline_accuracy}
    print(f"  Baseline (fp32): {baseline_accuracy:.1f}%")

    if soft_tokens is not None:
        # Test different quantization levels
        quantization_configs = [
            ("fp16", lambda x: x.half().float()),
            ("bf16", lambda x: x.bfloat16().float()),
            ("int8", lambda x: (x * 127).round().clamp(-128, 127) / 127),
            ("int4", lambda x: (x * 7).round().clamp(-8, 7) / 7),
        ]

        for quant_name, quant_fn in quantization_configs:
            print(f"\nTesting {quant_name} quantization...")

            try:
                # Quantize soft tokens
                quantized_tokens = quant_fn(soft_tokens)

                # Evaluate with quantized tokens
                quant_results = evaluate_bridge(
                    bridge_results["bridge_model"],
                    eval_data=eval_data,
                    soft_tokens_override=quantized_tokens,
                    device=device,
                )

                # Compute error
                quant_error = torch.mean((soft_tokens - quantized_tokens) ** 2).item()

                results[quant_name] = {
                    "accuracy": quant_results["accuracy"],
                    "accuracy_drop": baseline_accuracy - quant_results["accuracy"],
                    "quantization_mse": quant_error,
                }

                print(f"  {quant_name}: {quant_results['accuracy']:.1f}% (drop: {results[quant_name]['accuracy_drop']:.1f}pp)")

            except Exception as e:
                results[quant_name] = {"error": str(e)}
                print(f"  Error: {e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "soft_token_quantization_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# EXPERIMENT: MULTI-TASK BRIDGE (Chelsea Finn)
# =============================================================================

def exp_multi_task_bridge(output_dir, samples_per_task=1000, eval_samples=200):
    """
    Train a single bridge on multiple tasks.
    Addresses: Chelsea Finn Q1 - "If you train on multiple tasks jointly, does a universal bridge emerge?"
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Multi-Task Bridge Training")
    print("="*70)

    results = {
        "experiment": "multi_task_bridge",
        "timestamp": datetime.now().isoformat(),
        "description": "Joint training on SST-2 and AG News"
    }

    if not BRIDGE_AVAILABLE:
        results["status"] = "skipped"
        results["reason"] = "Bridge training modules not available"
        return results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    sst2 = load_dataset("glue", "sst2", trust_remote_code=True)
    agnews = load_dataset("ag_news", trust_remote_code=True)

    # Prepare combined training data with task labels
    combined_train = []

    # Add SST-2 samples
    for i, item in enumerate(sst2["train"]):
        if i >= samples_per_task:
            break
        combined_train.append({
            "text": item["sentence"],
            "label": item["label"],
            "task": "sst2",
            "task_labels": ["negative", "positive"],
        })

    # Add AG News samples
    for i, item in enumerate(agnews["train"]):
        if i >= samples_per_task:
            break
        combined_train.append({
            "text": item["text"],
            "label": item["label"],
            "task": "agnews",
            "task_labels": ["World", "Sports", "Business", "Sci/Tech"],
        })

    # Shuffle combined data
    np.random.seed(42)
    np.random.shuffle(combined_train)

    print(f"Combined training data: {len(combined_train)} samples")
    print(f"  SST-2: {sum(1 for x in combined_train if x['task'] == 'sst2')}")
    print(f"  AG News: {sum(1 for x in combined_train if x['task'] == 'agnews')}")

    # Train multi-task bridge
    try:
        multi_task_results = train_bridge(
            sender_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            receiver_model_id="mistralai/Mistral-7B-Instruct-v0.3",
            train_data=combined_train,
            eval_data=None,  # Will evaluate separately per task
            num_soft_tokens=16,  # More tokens for multi-task
            training_steps=4000,
            multi_task=True,
            device=device,
        )

        # Evaluate on each task separately
        for task_name, task_dataset, text_field, label_field, labels in [
            ("sst2", sst2["validation"], "sentence", "label", ["negative", "positive"]),
            ("agnews", agnews["test"], "text", "label", ["World", "Sports", "Business", "Sci/Tech"]),
        ]:
            task_eval = list(task_dataset)[:eval_samples]
            eval_results = evaluate_bridge(
                multi_task_results["bridge_model"],
                eval_data=task_eval,
                text_field=text_field,
                label_field=label_field,
                labels=labels,
                device=device,
            )

            results[f"multi_task_{task_name}"] = {
                "accuracy": eval_results["accuracy"],
            }
            print(f"  {task_name}: {eval_results['accuracy']:.1f}%")

        # Compare to single-task bridges
        results["comparison"] = {
            "sst2_single_task_paper": 96.7,
            "agnews_single_task_paper": 90.7,
            "sst2_multi_task": results.get("multi_task_sst2", {}).get("accuracy"),
            "agnews_multi_task": results.get("multi_task_agnews", {}).get("accuracy"),
        }

    except Exception as e:
        results["error"] = str(e)
        print(f"Error: {e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "multi_task_bridge_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# PLACEHOLDER FOR DIRECT HIDDEN STATE BASELINE (Denny Zhou)
# =============================================================================

def exp_direct_hidden_state(output_dir):
    """
    Test direct hidden state transfer without cross-attention.
    Addresses: Denny Zhou Q5 - "What if you directly use Llama's hidden states?"

    Note: This requires modifying the bridge architecture to bypass cross-attention.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Direct Hidden State Baseline")
    print("="*70)

    results = {
        "experiment": "direct_hidden_state",
        "timestamp": datetime.now().isoformat(),
        "status": "placeholder",
        "description": "Direct hidden state projection without cross-attention compression",
        "implementation_notes": [
            "Requires bridge architecture modification",
            "Would use mean-pooled hidden states -> linear projection -> soft tokens",
            "Table 1 in paper shows 'Mean Pooling' achieves 0% (complete failure)",
            "This suggests direct hidden state transfer without cross-attention fails",
        ]
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "direct_hidden_state_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bridge-based reviewer experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "same_checkpoint", "minimum_data", "self_consistency",
                                "quantization", "multi_task", "direct_hidden"],
                        help="Which experiment to run")
    parser.add_argument("--output_dir", type=str, default="runs/bridge_reviewer",
                        help="Output directory")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="Evaluation samples")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Bridge modules available: {BRIDGE_AVAILABLE}")

    experiments = {
        "same_checkpoint": lambda: exp_same_checkpoint_baseline(output_dir),
        "minimum_data": lambda: exp_minimum_data(output_dir, eval_samples=args.eval_samples),
        "self_consistency": lambda: exp_self_consistency(output_dir, eval_samples=args.eval_samples),
        "quantization": lambda: exp_soft_token_quantization(output_dir, eval_samples=args.eval_samples),
        "multi_task": lambda: exp_multi_task_bridge(output_dir, eval_samples=args.eval_samples),
        "direct_hidden": lambda: exp_direct_hidden_state(output_dir),
    }

    all_results = {}

    if args.experiment == "all":
        for name, func in experiments.items():
            try:
                print(f"\n{'#'*70}")
                print(f"# {name.upper()}")
                print(f"{'#'*70}")
                all_results[name] = func()
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                all_results[name] = {"error": str(e)}
    else:
        all_results[args.experiment] = experiments[args.experiment]()

    # Save combined results
    with open(os.path.join(output_dir, "all_bridge_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
