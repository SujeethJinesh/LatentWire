#!/usr/bin/env python
# telepathy/train_telepathy_passkey.py
"""
Phase 20: Passkey Retrieval (Sanity Check)

GOAL: Test if the bridge can transmit PRECISE alphanumeric information.
This is a prerequisite for Math (GSM8K). If we can't copy a 5-digit code,
we certainly can't reason about numbers.

Setup:
- Source: "The secret access code is <12345>. Remember it."
- Target Primer: "The code is"
- Target Label: "<12345>"

Interpretation:
- 0% accuracy: Bridge is just a "Topic Classifier" - cannot transmit data
- 50-80%: Bridge is lossy - gets some digits, explains GSM8K failure
- >99%: High-fidelity data channel - GSM8K failure is reasoning/curriculum issue
"""
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Import the actual bridge
from latent_bridge_v15 import LatentBridgeV15


class PasskeyDataset(Dataset):
    """Generates synthetic passkey examples on the fly."""
    def __init__(self, size=5000, seed=None):
        self.size = size
        if seed is not None:
            random.seed(seed)
        self.data = [self._generate() for _ in range(size)]

    def _generate(self):
        # Generate random 5-digit code
        code = f"{random.randint(10000, 99999)}"
        return {
            "source_text": f"The secret access code is {code}. Remember it.",
            "target_label": code
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class Args:
    """Simple args object to match LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=16, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def get_nearest_neighbors(latent_vector, embedding_matrix, tokenizer, k=5):
    """Find k nearest vocabulary tokens to a latent vector."""
    latent_vector = latent_vector.float()
    embedding_matrix = embedding_matrix.float()

    latent_norm = F.normalize(latent_vector.unsqueeze(0), p=2, dim=-1)
    emb_norm = F.normalize(embedding_matrix, p=2, dim=-1)
    similarity = torch.matmul(latent_norm, emb_norm.t())

    scores, indices = torch.topk(similarity, k)

    neighbors = []
    for score, idx in zip(scores[0], indices[0]):
        token_str = tokenizer.decode([idx.item()]).replace('\n', '\\n').replace('\t', '\\t')
        if token_str.strip() == '':
            token_str = repr(tokenizer.decode([idx.item()]))
        neighbors.append((token_str, score.item()))
    return neighbors


def analyze_latent_interpretability(bridge, src_model, tgt_model, src_tok, tgt_tok, src_device, tgt_device, sample_texts, num_tokens):
    """Analyze what the soft tokens 'mean' by finding nearest vocabulary neighbors."""
    print("\n" + "=" * 70)
    print("LATENT INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print("What vocabulary tokens are closest to each soft token?")

    bridge.eval()
    mistral_embeddings = tgt_model.get_input_embeddings().weight.detach()

    for text in sample_texts:
        print(f"\n--- Input: \"{text[:60]}...\" ---")

        src_inputs = src_tok(text, return_tensors="pt").to(src_device)
        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]
            src_mask = src_inputs.attention_mask
            latents, _, _, _ = bridge(src_hidden.to(tgt_device), src_mask.to(tgt_device))

        latents = latents[0]  # Remove batch dim

        for i in range(min(num_tokens, latents.shape[0])):
            neighbors = get_nearest_neighbors(latents[i], mistral_embeddings, tgt_tok, k=5)
            neighbor_str = ", ".join([f"'{tok}'({score:.2f})" for tok, score in neighbors])
            print(f"  Token {i+1}: {neighbor_str}")

    # Geometry analysis
    print("\n--- Latent Geometry (last sample) ---")
    latents_norm = F.normalize(latents.float(), dim=-1)
    sim_matrix = torch.matmul(latents_norm, latents_norm.t())
    off_diag = sim_matrix[~torch.eye(num_tokens, dtype=torch.bool, device=tgt_device)]
    print(f"  Mean pairwise similarity: {off_diag.mean().item():.3f}")
    print(f"  Token RMS range: {latents.float().pow(2).mean(dim=-1).sqrt().min().item():.4f} - {latents.float().pow(2).mean(dim=-1).sqrt().max().item():.4f}")


def evaluate(bridge, src_model, tgt_model, src_tok, tgt_tok, src_device, tgt_device, num_samples=100, verbose=True):
    """Run exact match evaluation."""
    bridge.eval()
    correct = 0
    digit_correct = 0
    total_digits = 0
    total = 0

    if verbose:
        print("\n--- Evaluating Passkey Retrieval ---")

    # Generate fresh test set with fixed seed for reproducibility
    test_set = PasskeyDataset(size=num_samples, seed=42)

    primer = "The code is "
    primer_tokens = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(tgt_device)

    # Pre-compute primer embeddings
    with torch.no_grad():
        primer_embeds = tgt_model.get_input_embeddings()(primer_tokens.input_ids)

    results = []

    for i in range(num_samples):
        item = test_set[i]
        src_text = item['source_text']
        target = item['target_label']

        # 1. Source Forward (on src_device)
        src_inputs = src_tok(src_text, return_tensors="pt").to(src_device)
        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]  # Layer 31
            src_mask = src_inputs.attention_mask

        # 2. Bridge Forward (move hidden states to tgt_device)
        with torch.no_grad():
            # Bridge returns (latents, aux_loss, diversity, z_variance)
            latents, _, _, _ = bridge(src_hidden.to(tgt_device), src_mask.to(tgt_device))

        # 3. Target Generation
        # Input: [Primer] + [Latents]
        combined_embeds = torch.cat([primer_embeds, latents], dim=1)

        # Create attention mask (all ones - all tokens are valid)
        attn_mask = torch.ones(1, combined_embeds.shape[1], device=tgt_device)

        # Generate
        out_ids = tgt_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            max_new_tokens=10,
            pad_token_id=tgt_tok.eos_token_id,
            do_sample=False  # Greedy decoding for precision
        )

        # Decode
        output_text = tgt_tok.decode(out_ids[0], skip_special_tokens=True)

        # Extract digits from output
        pred_digits = ''.join(c for c in output_text if c.isdigit())

        # Check exact match
        is_exact = target in output_text or pred_digits == target
        if is_exact:
            correct += 1

        # Check digit-level accuracy
        for j, char in enumerate(target):
            total_digits += 1
            if j < len(pred_digits) and pred_digits[j] == char:
                digit_correct += 1

        results.append({
            "target": target,
            "pred": pred_digits[:5] if pred_digits else "NONE",
            "raw": output_text[:50],
            "exact": is_exact
        })
        total += 1

    acc = (correct / total) * 100
    digit_acc = (digit_correct / total_digits) * 100 if total_digits > 0 else 0

    if verbose:
        print(f"Exact Match: {acc:.1f}% ({correct}/{total})")
        print(f"Digit-Level: {digit_acc:.1f}% ({digit_correct}/{total_digits})")
        print("\nSample Outputs:")
        for res in results[:10]:
            status = "✓" if res['exact'] else "✗"
            print(f"  {status} GT: {res['target']} | Pred: {res['pred']} | Raw: {res['raw']}")

    return {
        "exact_match": acc,
        "digit_accuracy": digit_acc,
        "correct": correct,
        "total": total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/passkey_check")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--soft_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    args = parser.parse_args()

    # Multi-GPU: Llama on GPU 0, Mistral on GPU 1
    num_gpus = torch.cuda.device_count()
    src_device = torch.device("cuda:0") if num_gpus > 0 else torch.device("cpu")
    tgt_device = torch.device("cuda:1") if num_gpus > 1 else src_device
    print(f"Using {num_gpus} GPUs: Llama on {src_device}, Mistral on {tgt_device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PASSKEY RETRIEVAL EXPERIMENT")
    print("=" * 60)
    print(f"Goal: Can the bridge transmit a 5-digit code exactly?")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Steps: {args.steps}")
    print("=" * 60)

    # Load Models on separate GPUs
    print("\nLoading models...")
    src_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token
    src_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(src_device)
    src_model.eval()

    tgt_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16
    ).to(tgt_device)
    tgt_model.eval()

    # Freeze LLMs
    for p in src_model.parameters():
        p.requires_grad = False
    for p in tgt_model.parameters():
        p.requires_grad = False

    # Init Bridge with args object (on target device for Mistral interface)
    bridge_args = Args(
        soft_tokens=args.soft_tokens,
        heads=8,
        depth=2,
        use_fsq=False  # Continuous mode
    )
    bridge = LatentBridgeV15(
        bridge_args,
        src_dim=4096,
        tgt_dim=4096,
        target_rms=0.03
    ).to(tgt_device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)

    # Dataset
    ds = PasskeyDataset(size=args.steps * args.batch_size * 2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Primer embeddings (on target device)
    primer = "The code is "
    primer_tokens = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(tgt_device)
    with torch.no_grad():
        primer_embeds_single = tgt_model.get_input_embeddings()(primer_tokens.input_ids)
        primer_embeds = primer_embeds_single.repeat(args.batch_size, 1, 1)

    # Training Loop
    bridge.train()
    pbar = tqdm(range(args.steps), desc="Training")
    iter_loader = iter(loader)

    metrics_log = []

    for step in pbar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        src_texts = batch['source_text']
        targets = batch['target_label']

        B = len(src_texts)

        # 1. Get Source Hidden (on src_device)
        src_inputs = src_tok(
            list(src_texts),
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(src_device)

        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]
            src_mask = src_inputs.attention_mask

        # 2. Bridge Forward (move hidden states to tgt_device)
        latents, aux_loss, diversity, z_var = bridge(src_hidden.to(tgt_device), src_mask.to(tgt_device))

        # 3. Compute Diversity Loss (prevent mode collapse)
        flat_tokens = latents.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=tgt_device)
        off_diag_sim = sim_matrix[mask].mean()
        div_loss = off_diag_sim

        # 4. Target Forward - LM Loss (on tgt_device)
        target_inputs = tgt_tok(
            list(targets),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(tgt_device)
        target_embeds = tgt_model.get_input_embeddings()(target_inputs.input_ids)

        # Adjust primer for actual batch size
        primer_batch = primer_embeds[:B]

        # Concat inputs: [Primer] + [Latents] + [Target]
        inputs_embeds = torch.cat([primer_batch, latents, target_embeds], dim=1)

        # Create Labels: -100 for primer and latents, actual tokens for target (on tgt_device)
        ignore_len = primer_batch.shape[1] + latents.shape[1]
        labels = torch.full((B, inputs_embeds.shape[1]), -100, dtype=torch.long, device=tgt_device)
        labels[:, ignore_len:] = target_inputs.input_ids

        # Attention mask (on tgt_device)
        attn_mask = torch.ones(B, inputs_embeds.shape[1], device=tgt_device)

        # Mistral Forward
        outputs = tgt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels
        )
        lm_loss = outputs.loss

        # Total loss
        loss = lm_loss + args.diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({
            'lm': f"{lm_loss.item():.3f}",
            'div': f"{div_loss.item():.3f}",
            'scale': f"{bridge.output_scale.item():.4f}"
        })

        # Periodic evaluation
        if (step + 1) % args.eval_every == 0:
            eval_results = evaluate(
                bridge, src_model, tgt_model, src_tok, tgt_tok, src_device, tgt_device,
                num_samples=50, verbose=True
            )
            metrics_log.append({
                "step": step + 1,
                "lm_loss": lm_loss.item(),
                "div_loss": div_loss.item(),
                **eval_results
            })
            bridge.train()

    # Final Evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (100 samples)")
    print("=" * 60)
    final_results = evaluate(
        bridge, src_model, tgt_model, src_tok, tgt_tok, src_device, tgt_device,
        num_samples=100, verbose=True
    )

    # Save checkpoint
    save_path = os.path.join(args.output_dir, "bridge_passkey.pt")
    torch.save(bridge.state_dict(), save_path)
    print(f"\nCheckpoint saved to: {save_path}")

    # Save results
    results = {
        "experiment": "passkey_retrieval",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "soft_tokens": args.soft_tokens,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "diversity_weight": args.diversity_weight
        },
        "final_results": final_results,
        "training_log": metrics_log
    }

    results_path = os.path.join(args.output_dir, "passkey_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Latent Interpretability Analysis
    sample_texts = [
        "The secret access code is 12345. Remember it.",
        "The secret access code is 99999. Remember it.",
        "The secret access code is 54321. Remember it.",
    ]
    analyze_latent_interpretability(
        bridge, src_model, tgt_model, src_tok, tgt_tok, src_device, tgt_device,
        sample_texts, args.soft_tokens
    )

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    acc = final_results["exact_match"]
    if acc < 10:
        print("❌ FAILURE: Bridge is a 'Vibe Transfer' - cannot transmit precise data.")
        print("   → GSM8K failure was due to BANDWIDTH, not reasoning.")
        print("   → Need architectural changes for precise information transfer.")
    elif acc < 80:
        print("⚠️  PARTIAL: Bridge is lossy - some digits transfer, some don't.")
        print("   → GSM8K failure was partially bandwidth-limited.")
        print("   → Try increasing soft_tokens or different architecture.")
    else:
        print("✅ SUCCESS: Bridge is a high-fidelity data channel!")
        print("   → GSM8K failure was due to REASONING/CURRICULUM, not bandwidth.")
        print("   → Try COCONUT-style curriculum learning for math.")


if __name__ == "__main__":
    main()
