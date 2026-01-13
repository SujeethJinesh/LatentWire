#!/usr/bin/env python
"""
Reasoning Hail Mary Experiments
================================
A consolidated suite of quick experiments to explore reasoning transfer.

This script runs multiple small experiments in sequence to identify
promising directions for reasoning via soft tokens.

Experiments:
1. Token Capacity Scaling (M=8,16,32,64 on Curriculum Arithmetic)
2. Difficulty Progression (train on easy, test on hard)
3. CoT Prompt Engineering (receiver-side "Let's think step by step")
4. Longer Training (5000 steps vs default 2000)
5. Linear Probe Upper Bound (can Llama's hidden states predict answers?)

Total runtime target: ~4-6 hours on single H100
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from latentwire import LatentBridge
from latentwire.data import (
    load_curriculum_arithmetic_subset,
    load_gsm8k_subset,
)


class ExperimentConfig:
    """Configuration for a single experiment."""
    def __init__(self, name: str, description: str, **kwargs):
        self.name = name
        self.description = description
        self.params = kwargs


def setup_models(device: torch.device, bf16: bool = True):
    """Load Llama (sender) and Mistral (receiver) models."""
    print("Loading Llama 3.1 8B (sender)...")
    src_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map={"": device}
    ).eval()

    print("Loading Mistral 7B (receiver)...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map={"": device}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    src_tok.pad_token = src_tok.eos_token

    tgt_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tgt_tok.pad_token = tgt_tok.eos_token

    return src_model, tgt_model, src_tok, tgt_tok


def compute_target_rms(tgt_model) -> float:
    """Compute RMS of target model embeddings for normalization."""
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        return tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()


def train_bridge(
    bridge: LatentBridge,
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    train_data: List[Dict],
    device: torch.device,
    steps: int = 1000,
    batch_size: int = 8,
    lr: float = 2e-4,
    source_layer: int = 31,
    cot_prefix: str = None,  # If set, prepend this to receiver
    bf16: bool = True,
    progress_desc: str = "Training"
) -> Dict[str, Any]:
    """Train bridge on arithmetic data."""
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)

    # Build batches
    texts = [ex["source"] for ex in train_data]
    answers = [ex["answer"] for ex in train_data]

    losses = []
    pbar = tqdm(range(steps), desc=progress_desc, ncols=100)

    for step in pbar:
        # Sample batch
        idxs = torch.randint(0, len(texts), (batch_size,)).tolist()
        batch_texts = [texts[i] for i in idxs]
        batch_answers = [answers[i] for i in idxs]

        optimizer.zero_grad()

        # Encode with Llama
        with torch.no_grad():
            src_enc = src_tok(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer]
            if bf16:
                src_h = src_h.bfloat16()

        # Generate soft tokens
        soft_tokens, aux_loss, diversity, z_var = bridge(src_h, src_enc.attention_mask)

        # Prepare target
        B = soft_tokens.shape[0]
        K = soft_tokens.shape[1]

        # Optional CoT prefix
        if cot_prefix:
            prefix_text = cot_prefix
        else:
            prefix_text = "Answer:"

        with torch.no_grad():
            prefix_enc = tgt_tok(
                [prefix_text] * B, return_tensors="pt", add_special_tokens=False
            ).to(device)
            prefix_embeds = tgt_model.get_input_embeddings()(prefix_enc.input_ids)
            if bf16:
                prefix_embeds = prefix_embeds.bfloat16()

            # Target answers
            tgt_texts = [f" {a}{tgt_tok.eos_token}" for a in batch_answers]
            tgt_enc = tgt_tok(
                tgt_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=32, add_special_tokens=False
            ).to(device)
            answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
            if bf16:
                answer_embeds = answer_embeds.bfloat16()

        # Concat: [prefix] + [soft_tokens] + [answer]
        inputs_embeds = torch.cat([prefix_embeds, soft_tokens, answer_embeds], dim=1)

        # Labels: mask prefix and soft tokens
        P_len = prefix_embeds.shape[1]
        ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
        answer_labels = tgt_enc.input_ids.clone()
        answer_labels[tgt_enc.attention_mask == 0] = -100
        labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

        # Attention mask
        soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
        full_mask = torch.cat([prefix_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

        # Forward
        outputs = tgt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels_tensor
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    return {
        "final_loss": losses[-1] if losses else float("inf"),
        "avg_loss": sum(losses) / len(losses) if losses else float("inf"),
        "losses": losses[-100:]  # Keep last 100
    }


def evaluate_bridge(
    bridge: LatentBridge,
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    eval_data: List[Dict],
    device: torch.device,
    source_layer: int = 31,
    cot_prefix: str = None,
    bf16: bool = True,
    max_new_tokens: int = 20,
    progress_desc: str = "Evaluating"
) -> Dict[str, Any]:
    """Evaluate bridge on test data."""
    bridge.eval()

    correct = 0
    total = 0
    predictions = []

    for ex in tqdm(eval_data, desc=progress_desc, ncols=100):
        text = ex["source"]
        answer = ex["answer"]

        # Extract numeric answer if GSM8K style
        if "####" in answer:
            answer = answer.split("####")[-1].strip()

        with torch.no_grad():
            src_enc = src_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer]
            if bf16:
                src_h = src_h.bfloat16()

            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Prepare generation
            if cot_prefix:
                prefix_text = cot_prefix
            else:
                prefix_text = "Answer:"

            prefix_enc = tgt_tok(prefix_text, return_tensors="pt", add_special_tokens=False).to(device)
            prefix_embeds = tgt_model.get_input_embeddings()(prefix_enc.input_ids)
            if bf16:
                prefix_embeds = prefix_embeds.bfloat16()

            combined_embeds = torch.cat([prefix_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip()

        # Extract numeric prediction
        pred_nums = [s for s in output.split() if s.lstrip('-').replace('.','').isdigit()]
        pred = pred_nums[0] if pred_nums else output[:20]

        is_correct = str(answer).strip() == str(pred).strip()
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "input": text[:50],
            "gold": answer,
            "pred": pred,
            "full_output": output[:100],
            "correct": is_correct
        })

    accuracy = 100 * correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions[:20]  # Sample of predictions
    }


def run_linear_probe_baseline(
    src_model,
    src_tok,
    train_data: List[Dict],
    eval_data: List[Dict],
    device: torch.device,
    source_layer: int = 31,
    bf16: bool = True,
) -> Dict[str, Any]:
    """
    Linear probe baseline: Can a linear layer on Llama hidden states predict the answer?
    This establishes an upper bound on what information is available in the sender's representations.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Linear Probe Upper Bound")
    print("="*60)
    print("Testing if Llama's hidden states contain the answer information...")

    # For arithmetic, we'll do classification into answer buckets
    # Collect unique answers from training data
    train_answers = sorted(set(ex["answer"] for ex in train_data))
    if len(train_answers) > 100:
        # Too many unique answers, use top 100
        from collections import Counter
        counts = Counter(ex["answer"] for ex in train_data)
        train_answers = [a for a, _ in counts.most_common(100)]

    answer_to_idx = {a: i for i, a in enumerate(train_answers)}
    n_classes = len(train_answers)
    print(f"Number of answer classes: {n_classes}")

    # Simple linear probe
    hidden_size = src_model.config.hidden_size
    probe = torch.nn.Linear(hidden_size, n_classes).to(device)
    if bf16:
        probe = probe.bfloat16()

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    probe.train()
    for epoch in range(3):  # Quick 3 epochs
        total_loss = 0
        n_batches = 0

        for i in range(0, min(len(train_data), 1000), 8):
            batch = train_data[i:i+8]
            batch_texts = [ex["source"] for ex in batch]
            batch_answers = [ex["answer"] for ex in batch]

            # Filter to known answers
            valid_idxs = [j for j, a in enumerate(batch_answers) if a in answer_to_idx]
            if not valid_idxs:
                continue

            batch_texts = [batch_texts[j] for j in valid_idxs]
            labels = torch.tensor([answer_to_idx[batch_answers[j]] for j in valid_idxs], device=device)

            with torch.no_grad():
                enc = src_tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                out = src_model(**enc, output_hidden_states=True)
                h = out.hidden_states[source_layer]
                if bf16:
                    h = h.bfloat16()
                # Pool: mean over sequence
                h_pooled = h.mean(dim=1)

            optimizer.zero_grad()
            logits = probe(h_pooled)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  Epoch {epoch+1}: Loss = {total_loss/max(1,n_batches):.3f}")

    # Evaluation
    probe.eval()
    correct = 0
    total = 0

    for ex in eval_data[:200]:
        answer = ex["answer"]
        if answer not in answer_to_idx:
            continue

        with torch.no_grad():
            enc = src_tok(ex["source"], return_tensors="pt", truncation=True, max_length=256).to(device)
            out = src_model(**enc, output_hidden_states=True)
            h = out.hidden_states[source_layer].mean(dim=1)
            if bf16:
                h = h.bfloat16()
            logits = probe(h)
            pred_idx = logits.argmax(dim=-1).item()

        pred = train_answers[pred_idx] if pred_idx < len(train_answers) else ""
        if pred == answer:
            correct += 1
        total += 1

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Linear Probe Accuracy: {accuracy:.1f}% ({correct}/{total})")

    return {
        "name": "linear_probe_baseline",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "n_classes": n_classes,
        "note": "Upper bound: how much answer info is in Llama hidden states"
    }


def run_experiment_1_token_scaling(
    src_model, tgt_model, src_tok, tgt_tok, device, output_dir: Path, args
) -> Dict[str, Any]:
    """
    Experiment 1: Token Capacity Scaling
    Test if more soft tokens help with reasoning (M=8,16,32,64)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Token Capacity Scaling")
    print("="*60)
    print("Hypothesis: Reasoning requires more soft tokens than classification")

    target_rms = compute_target_rms(tgt_model)

    # Use curriculum arithmetic at medium difficulty
    train_data = load_curriculum_arithmetic_subset(
        split="train", samples=2000, seed=42, difficulty=3
    )
    eval_data = load_curriculum_arithmetic_subset(
        split="test", samples=200, seed=42, difficulty=3
    )

    token_configs = [8, 16, 32, 64]
    results = {}

    for M in token_configs:
        print(f"\n--- Testing M={M} soft tokens ---")

        # Create fresh bridge
        class Args:
            soft_tokens = M
            depth = 2
            heads = 8

        bridge = LatentBridge(
            Args(),
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms
        )
        if args.bf16:
            bridge = bridge.bfloat16()
        bridge.to(device)

        # Train
        train_result = train_bridge(
            bridge, src_model, tgt_model, src_tok, tgt_tok,
            train_data, device, steps=500, batch_size=8,
            progress_desc=f"Train M={M}"
        )

        # Eval
        eval_result = evaluate_bridge(
            bridge, src_model, tgt_model, src_tok, tgt_tok,
            eval_data, device, progress_desc=f"Eval M={M}"
        )

        results[M] = {
            "train": train_result,
            "eval": eval_result,
        }
        print(f"M={M}: Accuracy = {eval_result['accuracy']:.1f}%")

        # Save checkpoint
        torch.save(bridge.state_dict(), output_dir / f"exp1_bridge_M{M}.pt")

    return {
        "name": "token_scaling",
        "hypothesis": "More tokens improve reasoning",
        "results": results,
        "conclusion": analyze_scaling(results)
    }


def run_experiment_2_difficulty_transfer(
    src_model, tgt_model, src_tok, tgt_tok, device, output_dir: Path, args
) -> Dict[str, Any]:
    """
    Experiment 2: Difficulty Transfer
    Train on easy problems (difficulty 1-2), test on hard (difficulty 4-5)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Difficulty Transfer")
    print("="*60)
    print("Can a bridge trained on easy arithmetic generalize to harder problems?")

    target_rms = compute_target_rms(tgt_model)

    # Train on easy
    train_data = load_curriculum_arithmetic_subset(
        split="train", samples=2000, seed=42,
        min_difficulty=1, max_difficulty=2
    )

    # Test on multiple difficulties
    test_sets = {
        "easy (1-2)": load_curriculum_arithmetic_subset(
            split="test", samples=100, seed=42, min_difficulty=1, max_difficulty=2
        ),
        "medium (3)": load_curriculum_arithmetic_subset(
            split="test", samples=100, seed=42, difficulty=3
        ),
        "hard (4-5)": load_curriculum_arithmetic_subset(
            split="test", samples=100, seed=42, min_difficulty=4, max_difficulty=5
        ),
    }

    class Args:
        soft_tokens = 16
        depth = 2
        heads = 8

    bridge = LatentBridge(
        Args(),
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.to(device)

    # Train
    train_result = train_bridge(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        train_data, device, steps=1000, batch_size=8,
        progress_desc="Train easy"
    )

    # Evaluate on each difficulty
    results = {"train": train_result}
    for name, eval_data in test_sets.items():
        eval_result = evaluate_bridge(
            bridge, src_model, tgt_model, src_tok, tgt_tok,
            eval_data, device, progress_desc=f"Eval {name}"
        )
        results[name] = eval_result
        print(f"{name}: Accuracy = {eval_result['accuracy']:.1f}%")

    torch.save(bridge.state_dict(), output_dir / "exp2_bridge_easy_trained.pt")

    return {
        "name": "difficulty_transfer",
        "hypothesis": "Bridge trained on easy transfers to hard",
        "results": results
    }


def run_experiment_3_cot_prompting(
    src_model, tgt_model, src_tok, tgt_tok, device, output_dir: Path, args
) -> Dict[str, Any]:
    """
    Experiment 3: Chain-of-Thought Prompting
    Test if adding "Let's think step by step" helps the receiver
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: CoT Prompt Engineering")
    print("="*60)
    print("Does receiver-side 'Let me think step by step' help?")

    target_rms = compute_target_rms(tgt_model)

    train_data = load_curriculum_arithmetic_subset(
        split="train", samples=2000, seed=42, difficulty=3
    )
    eval_data = load_curriculum_arithmetic_subset(
        split="test", samples=200, seed=42, difficulty=3
    )

    prefixes = {
        "baseline": "Answer:",
        "cot_short": "Let me solve this. Answer:",
        "cot_long": "Let me think step by step.\n\nAnswer:",
    }

    results = {}

    for prefix_name, prefix_text in prefixes.items():
        print(f"\n--- Testing prefix: '{prefix_name}' ---")

        class Args:
            soft_tokens = 16
            depth = 2
            heads = 8

        bridge = LatentBridge(
            Args(),
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms
        )
        if args.bf16:
            bridge = bridge.bfloat16()
        bridge.to(device)

        # Train
        train_result = train_bridge(
            bridge, src_model, tgt_model, src_tok, tgt_tok,
            train_data, device, steps=500, batch_size=8,
            cot_prefix=prefix_text, progress_desc=f"Train {prefix_name}"
        )

        # Eval
        eval_result = evaluate_bridge(
            bridge, src_model, tgt_model, src_tok, tgt_tok,
            eval_data, device, cot_prefix=prefix_text,
            max_new_tokens=50,  # More tokens for CoT
            progress_desc=f"Eval {prefix_name}"
        )

        results[prefix_name] = {
            "prefix": prefix_text,
            "train": train_result,
            "eval": eval_result,
        }
        print(f"{prefix_name}: Accuracy = {eval_result['accuracy']:.1f}%")

        torch.save(bridge.state_dict(), output_dir / f"exp3_bridge_{prefix_name}.pt")

    return {
        "name": "cot_prompting",
        "hypothesis": "CoT prefix improves reasoning",
        "results": results
    }


def run_experiment_4_longer_training(
    src_model, tgt_model, src_tok, tgt_tok, device, output_dir: Path, args
) -> Dict[str, Any]:
    """
    Experiment 4: Longer Training
    Test if reasoning just needs more training steps
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Longer Training")
    print("="*60)
    print("Does reasoning improve with 5x more training steps?")

    target_rms = compute_target_rms(tgt_model)

    train_data = load_curriculum_arithmetic_subset(
        split="train", samples=5000, seed=42, difficulty=3
    )
    eval_data = load_curriculum_arithmetic_subset(
        split="test", samples=200, seed=42, difficulty=3
    )

    step_configs = [500, 1000, 2500, 5000]

    class Args:
        soft_tokens = 16
        depth = 2
        heads = 8

    bridge = LatentBridge(
        Args(),
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.to(device)

    results = {}
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=2e-4, weight_decay=0.01)

    # Build batches
    texts = [ex["source"] for ex in train_data]
    answers = [ex["answer"] for ex in train_data]

    current_step = 0
    pbar = tqdm(range(max(step_configs)), desc="Long training", ncols=100)

    for step in pbar:
        # Sample batch
        idxs = torch.randint(0, len(texts), (8,)).tolist()
        batch_texts = [texts[i] for i in idxs]
        batch_answers = [answers[i] for i in idxs]

        optimizer.zero_grad()

        with torch.no_grad():
            src_enc = src_tok(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[31]
            if args.bf16:
                src_h = src_h.bfloat16()

        soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)
        B = soft_tokens.shape[0]
        K = soft_tokens.shape[1]

        with torch.no_grad():
            prefix_enc = tgt_tok(["Answer:"] * B, return_tensors="pt", add_special_tokens=False).to(device)
            prefix_embeds = tgt_model.get_input_embeddings()(prefix_enc.input_ids)
            if args.bf16:
                prefix_embeds = prefix_embeds.bfloat16()

            tgt_texts = [f" {a}{tgt_tok.eos_token}" for a in batch_answers]
            tgt_enc = tgt_tok(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=32, add_special_tokens=False).to(device)
            answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
            if args.bf16:
                answer_embeds = answer_embeds.bfloat16()

        inputs_embeds = torch.cat([prefix_embeds, soft_tokens, answer_embeds], dim=1)
        P_len = prefix_embeds.shape[1]
        ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
        answer_labels = tgt_enc.input_ids.clone()
        answer_labels[tgt_enc.attention_mask == 0] = -100
        labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

        soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
        full_mask = torch.cat([prefix_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

        outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels_tensor)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        current_step = step + 1

        # Evaluate at checkpoints
        if current_step in step_configs:
            eval_result = evaluate_bridge(
                bridge, src_model, tgt_model, src_tok, tgt_tok,
                eval_data, device, progress_desc=f"Eval @ {current_step}"
            )
            results[current_step] = eval_result
            print(f"\nStep {current_step}: Accuracy = {eval_result['accuracy']:.1f}%")
            torch.save(bridge.state_dict(), output_dir / f"exp4_bridge_step{current_step}.pt")

    return {
        "name": "longer_training",
        "hypothesis": "More steps improve reasoning",
        "results": results
    }


def run_experiment_5_gsm8k_baseline(
    src_model, tgt_model, src_tok, tgt_tok, device, output_dir: Path, args
) -> Dict[str, Any]:
    """
    Experiment 5: GSM8K Baseline
    Test on real GSM8K math problems
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: GSM8K Baseline")
    print("="*60)
    print("Testing on real GSM8K math reasoning problems...")

    target_rms = compute_target_rms(tgt_model)

    train_data = load_gsm8k_subset(split="train", samples=2000, seed=42)
    eval_data = load_gsm8k_subset(split="test", samples=200, seed=42)

    class Args:
        soft_tokens = 32  # More tokens for complex reasoning
        depth = 2
        heads = 8

    bridge = LatentBridge(
        Args(),
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.to(device)

    # Train
    train_result = train_bridge(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        train_data, device, steps=2000, batch_size=8,
        progress_desc="Train GSM8K"
    )

    # Eval
    eval_result = evaluate_bridge(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        eval_data, device, max_new_tokens=50,
        progress_desc="Eval GSM8K"
    )

    print(f"GSM8K Accuracy: {eval_result['accuracy']:.1f}%")

    torch.save(bridge.state_dict(), output_dir / "exp5_bridge_gsm8k.pt")

    return {
        "name": "gsm8k_baseline",
        "hypothesis": "Test real math reasoning",
        "train": train_result,
        "eval": eval_result
    }


def analyze_scaling(results: Dict) -> str:
    """Analyze token scaling results."""
    accs = {M: results[M]["eval"]["accuracy"] for M in results}

    if max(accs.values()) < 10:
        return "FAILURE: All configurations below 10% accuracy"

    min_acc = min(accs.values())
    max_acc = max(accs.values())

    if max_acc - min_acc > 10:
        best_M = max(accs, key=accs.get)
        return f"SCALING: {best_M} tokens best ({max_acc:.1f}%), {max_acc - min_acc:.1f}% improvement over smallest"
    else:
        return f"PLATEAU: Token count doesn't strongly affect accuracy ({min_acc:.1f}% - {max_acc:.1f}%)"


def main():
    parser = argparse.ArgumentParser(description="Reasoning Hail Mary Experiments")
    parser.add_argument("--output_dir", default="runs/reasoning_hailmary")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--experiments", nargs="+", type=int, default=[1,2,3,4,5,6],
                       help="Which experiments to run (1-6)")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")

    # Load models
    src_model, tgt_model, src_tok, tgt_tok = setup_models(device, args.bf16)

    # Run experiments
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "experiments": {}
    }

    start_time = time.time()

    if 1 in args.experiments:
        result = run_experiment_1_token_scaling(
            src_model, tgt_model, src_tok, tgt_tok, device, output_dir, args
        )
        all_results["experiments"]["1_token_scaling"] = result

    if 2 in args.experiments:
        result = run_experiment_2_difficulty_transfer(
            src_model, tgt_model, src_tok, tgt_tok, device, output_dir, args
        )
        all_results["experiments"]["2_difficulty_transfer"] = result

    if 3 in args.experiments:
        result = run_experiment_3_cot_prompting(
            src_model, tgt_model, src_tok, tgt_tok, device, output_dir, args
        )
        all_results["experiments"]["3_cot_prompting"] = result

    if 4 in args.experiments:
        result = run_experiment_4_longer_training(
            src_model, tgt_model, src_tok, tgt_tok, device, output_dir, args
        )
        all_results["experiments"]["4_longer_training"] = result

    if 5 in args.experiments:
        result = run_experiment_5_gsm8k_baseline(
            src_model, tgt_model, src_tok, tgt_tok, device, output_dir, args
        )
        all_results["experiments"]["5_gsm8k_baseline"] = result

    if 6 in args.experiments:
        # Linear probe baseline
        train_data = load_curriculum_arithmetic_subset(split="train", samples=2000, seed=42, difficulty=3)
        eval_data = load_curriculum_arithmetic_subset(split="test", samples=200, seed=42, difficulty=3)
        result = run_linear_probe_baseline(
            src_model, src_tok, train_data, eval_data, device,
            source_layer=31, bf16=args.bf16
        )
        all_results["experiments"]["6_linear_probe"] = result

    # Summary
    total_time = time.time() - start_time
    all_results["total_time_seconds"] = total_time

    print("\n" + "="*70)
    print("REASONING HAIL MARY - FINAL SUMMARY")
    print("="*70)
    print(f"Total runtime: {total_time/3600:.1f} hours")
    print()

    for exp_name, exp_result in all_results["experiments"].items():
        print(f"\n{exp_name}:")
        if "results" in exp_result:
            for key, val in exp_result["results"].items():
                if isinstance(val, dict) and "accuracy" in val:
                    print(f"  {key}: {val['accuracy']:.1f}%")
                elif isinstance(val, dict) and "eval" in val:
                    print(f"  {key}: {val['eval']['accuracy']:.1f}%")
        elif "eval" in exp_result:
            print(f"  Accuracy: {exp_result['eval']['accuracy']:.1f}%")
        elif "accuracy" in exp_result:
            print(f"  Accuracy: {exp_result['accuracy']:.1f}%")

    # Save results
    results_path = output_dir / "all_results.json"

    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("="*70)


if __name__ == "__main__":
    main()
