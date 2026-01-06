#!/usr/bin/env python
# telepathy/train_telepathy_agnews.py
"""
Phase 17: AG News 4-Class Classification

Tests bridge on harder multi-class task after SST-2 success.
AG News: News article topic classification (4 classes)
- Class 0: World
- Class 1: Sports
- Class 2: Business
- Class 3: Sci/Tech

Uses optimal config from ablation study:
- Layer 31 (final layer has task-specific info)
- 8 soft tokens (information bottleneck principle)
- Continuous mode with diversity loss

Success Criteria:
- Random baseline: 25%
- If accuracy > 50%: Bridge works for multi-class
- If accuracy > 70%: Bridge is excellent
- If accuracy matches Mistral text baseline: Perfect transfer
"""
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from latent_bridge_v15 import LatentBridgeV15

# AG News class labels
AGNEWS_LABELS = ["world", "sports", "business", "science"]


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


def analyze_latent_interpretability(bridge, src_model, tgt_model, src_tok, tgt_tok, device, args, eval_ds):
    """Analyze what the soft tokens 'mean' by finding nearest vocabulary neighbors."""
    print("\n" + "=" * 70)
    print("LATENT INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print("What vocabulary tokens are closest to each soft token?")

    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()
    mistral_embeddings = tgt_model.get_input_embeddings().weight.detach()

    # Sample one from each class
    samples = []
    seen_labels = set()
    for i in range(min(200, len(eval_ds))):
        item = eval_ds[i]
        label = AGNEWS_LABELS[item['label']]
        if label not in seen_labels:
            samples.append((item['text'], label))
            seen_labels.add(label)
        if len(samples) >= 4:
            break

    for text, label in samples:
        print(f"\n--- Label: {label} ---")
        print(f"    Input: \"{text[:50]}...\"")

        src_input = f"Article: {text[:256]}\nTopic (world, sports, business, or science):"
        src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            latents, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

        latents = latents[0]  # Remove batch dim

        for i in range(min(args.soft_tokens, latents.shape[0])):
            neighbors = get_nearest_neighbors(latents[i], mistral_embeddings, tgt_tok, k=5)
            neighbor_str = ", ".join([f"'{tok}'({score:.2f})" for tok, score in neighbors])
            print(f"  Token {i+1}: {neighbor_str}")

    # Geometry analysis
    print("\n--- Latent Geometry (last sample) ---")
    latents_norm = F.normalize(latents.float(), dim=-1)
    sim_matrix = torch.matmul(latents_norm, latents_norm.t())
    num_tokens = latents.shape[0]
    off_diag = sim_matrix[~torch.eye(num_tokens, dtype=torch.bool, device=device)]
    print(f"  Mean pairwise similarity: {off_diag.mean().item():.3f}")
    print(f"  Token RMS range: {latents.float().pow(2).mean(dim=-1).sqrt().min().item():.4f} - {latents.float().pow(2).mean(dim=-1).sqrt().max().item():.4f}")

# Permissive matching for science/tech (AG News uses "Sci/Tech")
SCIENCE_SYNONYMS = ["science", "technology", "tech", "sci/tech", "scitech"]


def check_label_match(label, output):
    """Check if label matches output, with permissive matching for science."""
    if label == "science":
        return any(syn in output for syn in SCIENCE_SYNONYMS)
    return label in output


def setup_ddp():
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    # Optimal config from SST-2 ablation
    parser.add_argument("--source_layer", type=int, default=31)  # Final layer
    parser.add_argument("--soft_tokens", type=int, default=8)    # Information bottleneck
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)  # More steps for 4-class
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_path", default="bridge_agnews.pt")
    parser.add_argument("--output_dir", default="runs/agnews")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, device, args, step):
    """Quick evaluation during training."""
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()

    correct = 0
    total = 0
    class_correct = {l: 0 for l in AGNEWS_LABELS}
    class_total = {l: 0 for l in AGNEWS_LABELS}
    indices = list(range(0, min(100, len(eval_ds))))  # Eval 100 samples

    print(f"\n{'='*60}")
    print(f"QUICK EVAL @ Step {step}")
    print(f"{'='*60}")

    for i in indices:
        item = eval_ds[i]
        text = item['text']
        label = AGNEWS_LABELS[item['label']]

        # Source (same prompt format as baseline for fair comparison)
        src_input = f"Article: {text[:256]}\nTopic (world, sports, business, or science):"
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

            # Target: [Primer] + [Soft Tokens] -> Generate
            primer = "Topic:"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Check if correct label in output (permissive for science/tech)
        is_correct = check_label_match(label, output)
        if is_correct:
            correct += 1
            class_correct[label] += 1
        total += 1
        class_total[label] += 1

        # Print first 4 samples (one per class ideally)
        if i < 4:
            print(f"[{i}] GT: {label:8} | Pred: {output[:30]}")

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%) [Random=25%]")
    for label in AGNEWS_LABELS:
        if class_total[label] > 0:
            class_acc = 100 * class_correct[label] / class_total[label]
            print(f"  {label}: {class_correct[label]}/{class_total[label]} ({class_acc:.1f}%)")
    print(f"{'='*60}\n")

    bridge_module.train()
    return {"accuracy": accuracy, "correct": correct, "total": total}


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """Single training step."""
    # AG News Schema: {'text': str, 'label': int}
    inputs = batch['text']
    labels = [AGNEWS_LABELS[l] for l in batch['label']]

    B = len(inputs)

    # 1. Source (Llama reads article - same format as baseline)
    src_texts = [f"Article: {t[:256]}\nTopic (world, sports, business, or science):" for t in inputs]

    with torch.no_grad():
        src_enc = src_tok(
            src_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # 2. Bridge (continuous soft tokens)
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    soft_tokens, aux_loss, diversity, z_variance = bridge_module(src_h, src_mask)

    # 3. Batch diversity loss (prevent mode collapse)
    batch_div_loss = torch.tensor(0.0, device=device)
    if B > 1:
        flat_tokens = soft_tokens.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diag_sim = sim_matrix[mask].mean()
        batch_div_loss = off_diag_sim

    # 4. Target (Mistral predicts label)
    primer_text = "Topic:"
    with torch.no_grad():
        primer_enc = tgt_tok(
            [primer_text] * B, return_tensors="pt", add_special_tokens=False
        ).to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
        if args.bf16:
            primer_embeds = primer_embeds.bfloat16()

        tgt_texts = [f" {l}{tgt_tok.eos_token}" for l in labels]
        tgt_enc = tgt_tok(
            tgt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=16, add_special_tokens=False
        ).to(device)
        answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
        if args.bf16:
            answer_embeds = answer_embeds.bfloat16()

    # Concatenate: [Primer] + [Soft Tokens] + [Answer]
    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    K = soft_tokens.shape[1]
    P_len = primer_embeds.shape[1]

    # Labels: Mask primer and soft tokens, predict answer
    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100
    labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

    # Forward through Mistral
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels_tensor
    )
    loss_lm = outputs.loss

    # Total loss: LM + diversity penalty
    total_loss = loss_lm + args.diversity_weight * batch_div_loss

    return total_loss, {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "div": batch_div_loss.item(),
        "z_var": z_variance.item() if isinstance(z_variance, torch.Tensor) else z_variance
    }


def main():
    setup_ddp()
    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    # Create output directory
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Track training progress for JSON output
    training_log = []

    if local_rank == 0:
        print("=" * 60)
        print("Phase 17: AG News 4-Class Classification")
        print("=" * 60)
        print("")
        print("GOAL: Test bridge on harder multi-class task")
        print("  - Task: 4-class news topic classification")
        print("  - Classes: World, Sports, Business, Sci/Tech")
        print("  - Random baseline: 25%")
        print("")
        print("OPTIMAL CONFIG (from SST-2 ablation):")
        print(f"  - Source layer: {args.source_layer} (final layer)")
        print(f"  - Soft tokens: {args.soft_tokens} (information bottleneck)")
        print(f"  - Diversity weight: {args.diversity_weight}")
        print("")
        print(f"Training: {args.steps} steps, batch={args.batch_size}")
        print("=" * 60)

    # Load models
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": local_rank}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": local_rank}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
        if local_rank == 0:
            print(f"Target embedding RMS: {target_rms:.4f}")

    # Initialize bridge (CONTINUOUS, not VQ)
    bridge = LatentBridgeV15(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        bridge = DDP(bridge, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # Load AG News
    train_ds = load_dataset("ag_news", split="train")
    eval_ds = load_dataset("ag_news", split="test")

    if torch.distributed.is_initialized():
        train_ds = train_ds.shard(world_size, local_rank)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(train_ds)} samples")
        print(f"Test set: {len(eval_ds)} samples")
        print("Starting training...\n")

    progress = tqdm(range(args.steps), disable=(local_rank != 0), desc="AG News", ncols=100)
    iter_dl = iter(dl)
    running = {"total": 0, "lm": 0, "div": 0, "z_var": 0}
    grad_accum = args.grad_accum

    for step in progress:
        optimizer.zero_grad()
        accum_loss_dict = {"total": 0, "lm": 0, "div": 0, "z_var": 0}

        for _ in range(grad_accum):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)

            loss, loss_dict = train_step(
                batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args
            )

            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            for k in accum_loss_dict:
                accum_loss_dict[k] += loss_dict[k] / grad_accum

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        for k in running:
            running[k] += accum_loss_dict[k]

        progress.set_postfix({
            "lm": f"{accum_loss_dict['lm']:.2f}",
            "div": f"{accum_loss_dict['div']:.3f}"
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            avg = {k: v / 50 for k, v in running.items()}
            print(f"\n[Step {step+1}/{args.steps}]")
            print(f"  LM Loss: {avg['lm']:.3f}")
            print(f"  Batch Div Loss: {avg['div']:.4f}")
            print(f"  Z Variance: {avg['z_var']:.4f}")
            running = {k: 0 for k in running}

        # Quick eval
        if local_rank == 0 and (step + 1) % args.eval_every == 0:
            eval_result = quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, device, args, step + 1)
            training_log.append({"step": step + 1, **eval_result})

        # Save checkpoint
        if local_rank == 0 and (step + 1) % args.save_every == 0:
            bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
            torch.save(bridge_to_save.state_dict(), args.save_path)
            print(f"  Checkpoint saved: {args.save_path}")

    # Final save
    if local_rank == 0:
        bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
        torch.save(bridge_to_save.state_dict(), args.save_path)
        print("\n" + "=" * 60)
        print("Phase 17 AG News Training Complete!")
        print(f"Checkpoint: {args.save_path}")
        print("=" * 60)
        print("\nKEY METRICS:")
        print("  - Accuracy > 25%: Bridge transmits SOME info")
        print("  - Accuracy > 50%: Bridge works for multi-class")
        print("  - Accuracy > 70%: Bridge is excellent")
        print("  - Random baseline: 25%")

        # Final evaluation on more samples
        print("\n" + "=" * 60)
        print("FINAL EVALUATION (200 samples)")
        print("=" * 60)
        bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
        bridge_module.eval()
        correct = 0
        total = 0
        for i in range(min(200, len(eval_ds))):
            item = eval_ds[i]
            text = item['text']
            label = AGNEWS_LABELS[item['label']]
            src_input = f"Article: {text[:500]}\nTopic (world, sports, business, or science):"
            with torch.no_grad():
                src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
                src_out = src_model(**src_enc, output_hidden_states=True)
                src_h = src_out.hidden_states[args.source_layer]
                if args.bf16:
                    src_h = src_h.bfloat16()
                soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)
                primer = "Topic:"
                primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
                primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
                if args.bf16:
                    primer_embeds = primer_embeds.bfloat16()
                combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
                attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)
                out_ids = tgt_model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tgt_tok.eos_token_id,
                )
                output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()
            if check_label_match(label, output):
                correct += 1
            total += 1
        final_accuracy = 100 * correct / total
        print(f"Accuracy: {final_accuracy:.1f}% ({correct}/{total})")
        final_results = {"accuracy": final_accuracy, "correct": correct, "total": total}

        # Save JSON results
        results = {
            "experiment": "agnews",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "output_dir": args.output_dir,
                "steps": args.steps,
                "soft_tokens": args.soft_tokens,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "eval_every": args.eval_every,
                "diversity_weight": args.diversity_weight,
                "source_layer": args.source_layer,
            },
            "num_classes": 4,
            "final_results": final_results,
            "training_log": training_log
        }
        json_path = os.path.join(args.output_dir, "agnews_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")

        # Latent Interpretability Analysis
        analyze_latent_interpretability(bridge, src_model, tgt_model, src_tok, tgt_tok, device, args, eval_ds)


if __name__ == "__main__":
    main()
