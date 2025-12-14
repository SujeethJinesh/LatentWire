#!/usr/bin/env python
# telepathy/train_prompt_tuning_baseline.py
"""
CRITICAL BASELINE: Soft-Prompt Tuning on Mistral ONLY (No Llama)

This is the "killer experiment" for the paper:
- Train learnable soft tokens prepended to Mistral
- Same training budget (steps, data) as the bridge
- NO Llama involvement whatsoever

If Bridge >> Prompt-Tuning: Llama hidden states genuinely contribute
If Bridge ≈ Prompt-Tuning: Training helps, not Llama → super-additive claim invalid

This proves whether the sender model actually matters.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from datetime import datetime


class SoftPromptTuning(nn.Module):
    """
    Learnable soft prompts for a frozen LLM.

    This is the standard prompt-tuning baseline (Lester et al., 2021).
    No sender model, just learnable embeddings prepended to the receiver.
    """
    def __init__(self, num_tokens, embed_dim, target_rms=0.03):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Learnable soft prompt embeddings
        # Initialize with small random values
        self.soft_prompts = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

        # Output scale to match embedding magnitude
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        print(f"[SoftPromptTuning] {num_tokens} learnable tokens, dim={embed_dim}")
        print(f"  Parameters: {num_tokens * embed_dim:,}")

    def forward(self, batch_size):
        """
        Returns soft prompt embeddings for a batch.

        Args:
            batch_size: Number of examples in batch

        Returns:
            [B, num_tokens, embed_dim] soft prompts
        """
        # Expand for batch
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)

        # RMS normalize and scale (same as bridge)
        rms = torch.sqrt((prompts ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (prompts / rms) * self.output_scale

        return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset", default="sst2", choices=["sst2", "agnews", "trec", "rte"])
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--output_dir", default="runs/prompt_tuning_baseline")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
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
            "prompt_template": "Review: {text}\nSentiment (positive or negative):",
            "primer": "Sentiment:",
        },
        "agnews": {
            "load_args": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "num_classes": 4,
            "labels": ["World", "Sports", "Business", "Technology"],
            "prompt_template": "Article: {text}\nTopic:",
            "primer": "Topic:",
        },
        "trec": {
            "load_args": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "num_classes": 6,
            "labels": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
            "prompt_template": "Question: {text}\nType:",
            "primer": "Type:",
        },
        "rte": {
            "load_args": ("glue", "rte"),
            "text_field": None,  # Special handling for premise/hypothesis
            "label_field": "label",
            "num_classes": 2,
            "labels": ["entailment", "not_entailment"],
            "prompt_template": "Premise: {premise}\nHypothesis: {hypothesis}\nEntailment:",
            "primer": "Entailment:",
        },
    }
    return configs[dataset_name]


def format_example(item, config):
    """Format a single example according to dataset config."""
    if config["text_field"] is None:  # RTE special case
        return config["prompt_template"].format(
            premise=item["premise"],
            hypothesis=item["hypothesis"]
        )
    else:
        return config["prompt_template"].format(text=item[config["text_field"]])


def quick_eval(soft_prompt, model, tokenizer, eval_ds, config, device, args, step):
    """Quick evaluation during training."""
    soft_prompt.eval()

    correct = 0
    total = 0
    n_eval = min(50, len(eval_ds))

    print(f"\n{'='*60}")
    print(f"QUICK EVAL @ Step {step}")
    print(f"{'='*60}")

    for i in range(n_eval):
        item = eval_ds[i]
        label = config["labels"][item[config["label_field"]]]

        with torch.no_grad():
            # Get soft prompts
            prompts = soft_prompt(1)
            if args.bf16:
                prompts = prompts.bfloat16()

            # Get primer embeddings
            primer = config["primer"]
            primer_enc = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            # Combine: [Primer] + [Soft Prompts]
            combined_embeds = torch.cat([primer_embeds, prompts], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            # Generate
            out_ids = model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Check if correct
        if label.lower() in output.lower():
            correct += 1
        total += 1

        if i < 3:
            print(f"[{i}] GT: {label:15} | Pred: {output[:30]}")

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*60}\n")

    soft_prompt.train()
    return {"accuracy": accuracy, "correct": correct, "total": total}


def train_step(batch, tokenizer, soft_prompt, model, config, device, args):
    """Single training step."""
    labels_text = [config["labels"][l] for l in batch[config["label_field"]]]
    B = len(labels_text)

    # Get soft prompts
    prompts = soft_prompt(B)  # [B, K, D]
    if args.bf16:
        prompts = prompts.bfloat16()

    # Get primer embeddings
    primer = config["primer"]
    with torch.no_grad():
        primer_enc = tokenizer(
            [primer] * B, return_tensors="pt", add_special_tokens=False
        ).to(device)
        primer_embeds = model.get_input_embeddings()(primer_enc.input_ids)
        if args.bf16:
            primer_embeds = primer_embeds.bfloat16()

        # Get answer embeddings
        answer_texts = [f" {l}{tokenizer.eos_token}" for l in labels_text]
        answer_enc = tokenizer(
            answer_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=16, add_special_tokens=False
        ).to(device)
        answer_embeds = model.get_input_embeddings()(answer_enc.input_ids)
        if args.bf16:
            answer_embeds = answer_embeds.bfloat16()

    # Concatenate: [Primer] + [Soft Prompts] + [Answer]
    inputs_embeds = torch.cat([primer_embeds, prompts, answer_embeds], dim=1)

    K = prompts.shape[1]
    P_len = primer_embeds.shape[1]

    # Labels: Mask primer and soft prompts, predict answer
    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = answer_enc.input_ids.clone()
    answer_labels[answer_enc.attention_mask == 0] = -100
    labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, answer_enc.attention_mask], dim=1)

    # Forward
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels_tensor
    )

    return outputs.loss


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_dataset_config(args.dataset)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PROMPT-TUNING BASELINE (No Sender Model)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print("")
    print("This baseline proves whether Llama (sender) actually helps.")
    print("If Bridge >> Prompt-Tuning: Llama contributes meaningfully")
    print("If Bridge ≈ Prompt-Tuning: Only training helps, not Llama")
    print("=" * 60)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Compute target RMS
    with torch.no_grad():
        embeds = model.get_input_embeddings().weight.float()
        target_rms = embeds.pow(2).mean(dim=1).sqrt().median().item()
        print(f"Target embedding RMS: {target_rms:.4f}")

    # Initialize soft prompt tuning
    embed_dim = model.config.hidden_size
    soft_prompt = SoftPromptTuning(args.soft_tokens, embed_dim, target_rms)
    if args.bf16:
        soft_prompt = soft_prompt.bfloat16()
    soft_prompt.train()
    soft_prompt.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr=args.lr, weight_decay=0.01)

    # Load dataset
    train_ds = load_dataset(*config["load_args"], split="train")
    eval_split = "validation" if args.dataset in ["sst2", "rte"] else "test"
    eval_ds = load_dataset(*config["load_args"], split=eval_split)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"\nTraining on {len(train_ds)} samples")
    print(f"Validation: {len(eval_ds)} samples")
    print("Starting training...\n")

    training_log = []
    progress = tqdm(range(args.steps), desc=f"PromptTuning-{args.dataset}", ncols=100)
    iter_dl = iter(dl)
    running_loss = 0

    for step in progress:
        optimizer.zero_grad()
        accum_loss = 0

        for _ in range(args.grad_accum):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)

            loss = train_step(batch, tokenizer, soft_prompt, model, config, device, args)
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            accum_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
        optimizer.step()

        running_loss += accum_loss
        progress.set_postfix({"loss": f"{accum_loss:.3f}"})

        # Periodic logging
        if (step + 1) % 50 == 0:
            avg_loss = running_loss / 50
            print(f"\n[Step {step+1}/{args.steps}] Avg Loss: {avg_loss:.4f}")
            running_loss = 0

        # Quick eval
        if (step + 1) % args.eval_every == 0:
            eval_result = quick_eval(soft_prompt, model, tokenizer, eval_ds, config, device, args, step + 1)
            training_log.append({"step": step + 1, **eval_result})

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"prompt_tuning_{args.dataset}_step{step+1}.pt")
            torch.save(soft_prompt.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (200 samples)")
    print("=" * 60)

    soft_prompt.eval()
    correct = 0
    total = 0
    n_eval = min(200, len(eval_ds))

    for i in range(n_eval):
        item = eval_ds[i]
        label = config["labels"][item[config["label_field"]]]

        with torch.no_grad():
            prompts = soft_prompt(1)
            if args.bf16:
                prompts = prompts.bfloat16()

            primer = config["primer"]
            primer_enc = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, prompts], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        if label.lower() in output.lower():
            correct += 1
        total += 1

    final_accuracy = 100 * correct / total
    print(f"Final Accuracy: {final_accuracy:.1f}% ({correct}/{total})")

    # Save results
    results = {
        "experiment": f"prompt_tuning_baseline_{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "num_classes": config["num_classes"],
        "final_results": {"accuracy": final_accuracy, "correct": correct, "total": total},
        "training_log": training_log,
        "baseline_type": "prompt_tuning_no_sender"
    }

    json_path = os.path.join(args.output_dir, f"prompt_tuning_{args.dataset}_seed{args.seed}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Save final checkpoint
    final_ckpt = os.path.join(args.output_dir, f"prompt_tuning_{args.dataset}_final.pt")
    torch.save(soft_prompt.state_dict(), final_ckpt)
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
