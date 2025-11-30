#!/usr/bin/env python
# telepathy/eval_telepathy_v12.py
"""
Phase 12 Evaluation: Test Diffusion-Generated Soft Tokens

The diffusion bridge GENERATES soft tokens from noise, conditioned on source.
Unlike regression which outputs blurry averages, diffusion outputs should be
sharp vectors that lie ON the Mistral embedding manifold.

Evaluation:
1. Load trained diffusion bridge
2. Run Llama on question to get source hidden states
3. Sample from diffusion (Euler integration from noise)
4. Prepend generated soft tokens to Mistral
5. Generate answer and check entity preservation
"""
import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse

from latent_bridge_v12 import LatentBridgeV12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--diffusion_steps", type=int, default=10,
                        help="Number of Euler steps for diffusion sampling")
    parser.add_argument("--output_dir", default=".")
    return parser.parse_args()


def extract_entities(text):
    """Extract numbers, names, and nouns from text."""
    nums = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    caps = re.findall(r'\b[A-Z][a-z]+\b', text)
    return {
        'nums': list(set(nums[:10])),
        'names': list(set(caps[:10])),
    }


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Phase 12 Evaluation: Diffusion-Generated Soft Tokens")
    print("=" * 70)
    print(f"Diffusion steps: {args.diffusion_steps}")
    print("")
    print("Success Criteria:")
    print("  - Outputs VARY per input (not all identical)")
    print("  - Outputs contain question entities (Janet, ducks, numbers)")
    print("  - Entity transfer rate > 30% = Success!")
    print("=" * 70)
    print("")

    # Load models
    print("Loading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Load bridge
    bridge = LatentBridgeV12(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        num_latents=args.soft_tokens,
        depth=args.depth,
        heads=args.heads,
    ).bfloat16().to(DEVICE).eval()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    bridge.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load evaluation data
    ds = load_dataset("gsm8k", "main", split="test")
    indices = list(range(0, len(ds), len(ds) // args.num_samples))[:args.num_samples]

    print(f"\nEvaluating {len(indices)} samples...")
    print("-" * 70)

    results = []
    total_nums = 0
    matched_nums = 0
    total_names = 0
    matched_names = 0
    unique_outputs = set()

    for i, idx in enumerate(indices):
        sample = ds[idx]
        question = sample['question']

        # 1. Get source hidden states
        src_text = f"Question: {question}\nAnswer:"
        with torch.no_grad():
            src_enc = src_tok(src_text, return_tensors="pt").to(DEVICE)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer].bfloat16()
            src_mask = src_enc.attention_mask

        # 2. Sample soft tokens from diffusion
        with torch.no_grad():
            soft_tokens = bridge.sample(
                src_h, src_mask,
                num_steps=args.diffusion_steps
            )

        # 3. Add BOS and generate
        with torch.no_grad():
            bos_emb = tgt_model.get_input_embeddings()(
                torch.tensor([[tgt_tok.bos_token_id]], device=DEVICE)
            ).bfloat16()
            input_embeds = torch.cat([soft_tokens, bos_emb], dim=1)
            attn_mask = torch.ones(input_embeds.shape[:2], device=DEVICE, dtype=torch.long)

            gen_out = tgt_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id,
            )
            output_text = tgt_tok.decode(gen_out[0], skip_special_tokens=True)

        # Track unique outputs
        unique_outputs.add(output_text[:100])  # First 100 chars

        # Debug: show first sample
        if i == 0:
            print(f"\n[DEBUG] First sample raw output: '{output_text[:300]}'")
            print(f"[DEBUG] Generated {len(gen_out[0])} tokens")
            soft_stats = soft_tokens.float()
            print(f"[DEBUG] Soft token stats - min: {soft_stats.min():.4f}, "
                  f"max: {soft_stats.max():.4f}, mean: {soft_stats.mean():.4f}, "
                  f"std: {soft_stats.std():.4f}")
            print("")

        # Entity matching
        q_entities = extract_entities(question)
        o_entities = extract_entities(output_text)

        num_matches = len(set(q_entities['nums']) & set(o_entities['nums']))
        name_matches = len(set(q_entities['names']) & set(o_entities['names']))

        total_nums += len(q_entities['nums'])
        matched_nums += num_matches
        total_names += len(q_entities['names'])
        matched_names += name_matches

        results.append({
            'question': question[:80] + "...",
            'output': output_text[:200],
            'q_entities': q_entities,
            'o_entities': o_entities,
            'num_matches': num_matches,
            'name_matches': name_matches,
        })

        print(f"\n[Sample {idx}]")
        print(f"Q: {question[:80]}...")
        print(f"Q Entities: nums={q_entities['nums']}, names={q_entities['names']}")
        print(f"Output: {output_text[:100]}...")
        print(f"O Entities: nums={o_entities['nums']}, names={o_entities['names']}")
        print(f"Matches: nums={num_matches}/{len(q_entities['nums'])}, "
              f"names={name_matches}/{len(q_entities['names'])}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 12 EVALUATION SUMMARY")
    print("=" * 70)

    num_pct = 100 * matched_nums / max(total_nums, 1)
    name_pct = 100 * matched_names / max(total_names, 1)
    total_entities = total_nums + total_names
    total_matches = matched_nums + matched_names
    overall_pct = 100 * total_matches / max(total_entities, 1)
    unique_pct = 100 * len(unique_outputs) / args.num_samples

    print(f"Numbers: {matched_nums}/{total_nums} matched ({num_pct:.1f}%)")
    print(f"Names: {matched_names}/{total_names} matched ({name_pct:.1f}%)")
    print(f"\nOVERALL ENTITY TRANSFER: {total_matches}/{total_entities} ({overall_pct:.1f}%)")
    print(f"\nOUTPUT DIVERSITY: {len(unique_outputs)}/{args.num_samples} unique ({unique_pct:.1f}%)")
    print("")

    # Diagnosis
    if unique_pct < 50:
        print("=" * 70)
        print("LOW DIVERSITY: Outputs are still similar/identical.")
        print("Possible causes:")
        print("  1. Diffusion not trained enough (needs more steps)")
        print("  2. Conditioning too weak (source info not propagating)")
        print("  3. DiT architecture needs tuning")
        print("=" * 70)
    elif overall_pct < 20:
        print("=" * 70)
        print("LOW ENTITY TRANSFER but diverse outputs.")
        print("Bridge generates varied vectors but wrong content.")
        print("May need better source-target alignment during training.")
        print("=" * 70)
    elif overall_pct >= 30:
        print("=" * 70)
        print("SUCCESS! Entities are transferring through diffusion bridge.")
        print("Next: Increase diffusion steps, fine-tune hyperparameters.")
        print("=" * 70)
    else:
        print("=" * 70)
        print("PARTIAL SUCCESS: Some entity transfer, continue training.")
        print("=" * 70)

    # Save results
    output_path = os.path.join(args.output_dir, "eval_v12_results.json")
    summary = {
        'num_samples': args.num_samples,
        'diffusion_steps': args.diffusion_steps,
        'unique_outputs': len(unique_outputs),
        'unique_pct': unique_pct,
        'num_matches': matched_nums,
        'num_total': total_nums,
        'num_pct': num_pct,
        'name_matches': matched_names,
        'name_total': total_names,
        'name_pct': name_pct,
        'overall_matches': total_matches,
        'overall_total': total_entities,
        'overall_pct': overall_pct,
        'samples': results,
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
