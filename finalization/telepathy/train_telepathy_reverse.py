#!/usr/bin/env python
# telepathy/train_telepathy_reverse.py
"""
REVERSE DIRECTION: Mistral → Llama Bridge

Tests whether cross-model communication works bidirectionally.
- Sender: Mistral 7B (extracts hidden states)
- Receiver: Llama 3.1 8B (conditioned by soft tokens)

If both directions work: Architecture is general
If only Llama→Mistral works: Something specific about that direction
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from latent_bridge_v15 import LatentBridgeV15


def parse_args():
    parser = argparse.ArgumentParser()
    # REVERSED: Mistral is source, Llama is target
    parser.add_argument("--source_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--target_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="sst2", choices=["sst2", "agnews", "trec"])
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output_dir", default="runs/reverse_direction")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    return parser.parse_args()


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    configs = {
        "sst2": {
            "load_args": ("glue", "sst2"),
            "text_field": "sentence",
            "label_field": "label",
            "num_classes": 2,
            "labels": ["negative", "positive"],
            "src_template": "Review: {text}\nSentiment (positive or negative):",
            "primer": "Sentiment:",
        },
        "agnews": {
            "load_args": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "num_classes": 4,
            "labels": ["World", "Sports", "Business", "Technology"],
            "src_template": "Article: {text}\nTopic:",
            "primer": "Topic:",
        },
        "trec": {
            "load_args": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "num_classes": 6,
            "labels": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
            "src_template": "Question: {text}\nType:",
            "primer": "Type:",
        },
    }
    return configs[dataset_name]


def quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, config, device, args, step):
    """Quick evaluation during training."""
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()

    correct = 0
    total = 0
    n_eval = min(50, len(eval_ds))

    print(f"\n{'='*60}")
    print(f"QUICK EVAL @ Step {step} (REVERSE: Mistral→Llama)")
    print(f"{'='*60}")

    for i in range(n_eval):
        item = eval_ds[i]
        text = item[config["text_field"]]
        label = config["labels"][item[config["label_field"]]]

        src_input = config["src_template"].format(text=text)

        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

            primer = config["primer"]
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

        if label.lower() in output.lower():
            correct += 1
        total += 1

        if i < 3:
            print(f"[{i}] GT: {label:15} | Pred: {output[:30]}")

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*60}\n")

    bridge_module.train()
    return {"accuracy": accuracy, "correct": correct, "total": total}


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, config, device, args):
    """Single training step."""
    texts = batch[config["text_field"]]
    labels = [config["labels"][l] for l in batch[config["label_field"]]]
    B = len(texts)

    # Source (Mistral reads input)
    src_texts = [config["src_template"].format(text=t) for t in texts]

    with torch.no_grad():
        src_enc = src_tok(
            src_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        ).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # Bridge
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    soft_tokens, aux_loss, diversity, z_variance = bridge_module(src_h, src_mask)

    # Batch diversity loss
    batch_div_loss = torch.tensor(0.0, device=device)
    if B > 1:
        flat_tokens = soft_tokens.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diag_sim = sim_matrix[mask].mean()
        batch_div_loss = off_diag_sim

    # Target (Llama predicts label)
    primer = config["primer"]
    with torch.no_grad():
        primer_enc = tgt_tok(
            [primer] * B, return_tensors="pt", add_special_tokens=False
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

    # Concatenate
    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    K = soft_tokens.shape[1]
    P_len = primer_embeds.shape[1]

    # Labels
    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100
    labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

    # Forward
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels_tensor
    )
    loss_lm = outputs.loss

    total_loss = loss_lm + args.diversity_weight * batch_div_loss

    return total_loss, {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "div": batch_div_loss.item(),
        "z_var": z_variance.item() if isinstance(z_variance, torch.Tensor) else z_variance
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_dataset_config(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("REVERSE DIRECTION: Mistral → Llama Bridge")
    print("=" * 60)
    print(f"Source (sender): {args.source_model}")
    print(f"Target (receiver): {args.target_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Load models (REVERSED)
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto"
    ).eval()

    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto"
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS (for Llama now)
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
        print(f"Target (Llama) embedding RMS: {target_rms:.4f}")

    # Initialize bridge
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

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # Load dataset
    train_ds = load_dataset(*config["load_args"], split="train", trust_remote_code=True)
    eval_split = "validation" if args.dataset == "sst2" else "test"
    eval_ds = load_dataset(*config["load_args"], split=eval_split, trust_remote_code=True)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"\nTraining on {len(train_ds)} samples")
    print("Starting training...\n")

    training_log = []
    progress = tqdm(range(args.steps), desc=f"Reverse-{args.dataset}", ncols=100)
    iter_dl = iter(dl)
    running = {"total": 0, "lm": 0, "div": 0, "z_var": 0}

    for step in progress:
        optimizer.zero_grad()
        accum_loss_dict = {"total": 0, "lm": 0, "div": 0, "z_var": 0}

        for _ in range(args.grad_accum):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)

            loss, loss_dict = train_step(
                batch, src_tok, tgt_tok, src_model, bridge, tgt_model, config, device, args
            )

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            for k in accum_loss_dict:
                accum_loss_dict[k] += loss_dict[k] / args.grad_accum

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        for k in running:
            running[k] += accum_loss_dict[k]

        progress.set_postfix({"lm": f"{accum_loss_dict['lm']:.2f}"})

        if (step + 1) % args.eval_every == 0:
            eval_result = quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, config, device, args, step + 1)
            training_log.append({"step": step + 1, **eval_result})

        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"reverse_{args.dataset}_step{step+1}.pt")
            torch.save(bridge.state_dict(), ckpt_path)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (200 samples) - REVERSE DIRECTION")
    print("=" * 60)

    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()
    correct = 0
    total = 0

    for i in range(min(200, len(eval_ds))):
        item = eval_ds[i]
        text = item[config["text_field"]]
        label = config["labels"][item[config["label_field"]]]

        src_input = config["src_template"].format(text=text)

        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

            primer = config["primer"]
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

        if label.lower() in output.lower():
            correct += 1
        total += 1

    final_accuracy = 100 * correct / total
    print(f"Final Accuracy: {final_accuracy:.1f}% ({correct}/{total})")

    # Save results
    results = {
        "experiment": f"reverse_direction_{args.dataset}",
        "direction": "mistral_to_llama",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "num_classes": config["num_classes"],
        "final_results": {"accuracy": final_accuracy, "correct": correct, "total": total},
        "training_log": training_log
    }

    json_path = os.path.join(args.output_dir, f"reverse_{args.dataset}_seed{args.seed}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
