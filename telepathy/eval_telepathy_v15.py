#!/usr/bin/env python
# telepathy/eval_telepathy_v15.py
"""
Phase 15 Evaluation: VQ-Telepathy

Tests if discrete bottleneck produces coherent, relevant outputs.

Success Criteria:
1. Outputs are coherent (not garbage/repetitive)
2. Outputs mention relevant entities from question
3. Entity transfer rate > 30%
"""
import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

from latent_bridge_v15 import LatentBridgeV15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def extract_answer(text):
    """Extract numerical answer from GSM8K format."""
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    matches = re.findall(r'\d+', text)
    if matches:
        return matches[-1]
    return "[None]"


def extract_entities(text):
    """Extract numbers, names from text for entity matching."""
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
    print("Phase 15 Evaluation: VQ-Telepathy")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print("")
    print("VQ-Telepathy Key Features:")
    print("  - Discrete bottleneck (4096 codebook entries)")
    print("  - Prevents blur/drift via quantization")
    print("  - 1-step inference (no diffusion)")
    print("")
    print("Success Criteria:")
    print("  - Coherent outputs (not repetitive garbage)")
    print("  - Entity transfer rate > 30%")
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

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Create args-like object for bridge init
    class BridgeArgs:
        pass
    bridge_args = BridgeArgs()
    bridge_args.stats_path = args.stats_path
    bridge_args.soft_tokens = args.soft_tokens
    bridge_args.depth = args.depth
    bridge_args.heads = args.heads

    # Load bridge
    bridge = LatentBridgeV15(
        bridge_args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    bridge.load_state_dict(checkpoint)
    bridge.to(DEVICE)
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load evaluation data
    ds = load_dataset("gsm8k", "main", split="test")
    indices = list(range(0, min(len(ds), args.num_samples * 10), 10))[:args.num_samples]

    print(f"\nEvaluating {len(indices)} samples...")
    print("-" * 70)

    results = []
    total_nums = 0
    matched_nums = 0
    total_names = 0
    matched_names = 0
    unique_outputs = set()
    correct_answers = 0

    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        sample = ds[idx]
        question = sample['question']
        gt_answer = extract_answer(sample['answer'])

        # 1. Get source hidden states from Llama
        src_text = f"Question: {question}\nAnswer:"
        with torch.no_grad():
            src_enc = src_tok(src_text, return_tensors="pt").to(DEVICE)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            src_mask = src_enc.attention_mask

        # 2. Get quantized soft tokens from bridge
        with torch.no_grad():
            soft_tokens, vq_loss, perplexity = bridge(src_h, src_mask)

        # 3. Create input for Mistral
        primer = "Answer: "
        with torch.no_grad():
            primer_enc = tgt_tok(primer, return_tensors="pt").to(DEVICE)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            # [soft_tokens] + [primer] -> generate
            combined_embeds = torch.cat([soft_tokens, primer_embeds], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=DEVICE, dtype=torch.long)

            gen_out = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id,
            )
            output_text = tgt_tok.decode(gen_out[0], skip_special_tokens=True)

        # Track unique outputs
        unique_outputs.add(output_text[:100])

        # Extract predicted answer
        pred_answer = extract_answer(output_text)
        is_correct = (pred_answer == gt_answer)
        if is_correct:
            correct_answers += 1

        # Debug: show first sample
        if i == 0:
            print(f"\n[DEBUG] First sample raw output: '{output_text[:300]}'")
            print(f"[DEBUG] Generated {len(gen_out[0])} tokens")
            soft_stats = soft_tokens.float()
            print(f"[DEBUG] Soft token stats - min: {soft_stats.min():.4f}, "
                  f"max: {soft_stats.max():.4f}, mean: {soft_stats.mean():.4f}, "
                  f"std: {soft_stats.std():.4f}")
            print(f"[DEBUG] VQ perplexity: {perplexity:.1f}")
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
            'id': idx,
            'question': question[:80] + "...",
            'output': output_text[:200],
            'pred_answer': pred_answer,
            'gt_answer': gt_answer,
            'correct': is_correct,
            'q_entities': q_entities,
            'o_entities': o_entities,
            'num_matches': num_matches,
            'name_matches': name_matches,
        })

        # Print sample details
        print(f"\n[Sample {idx}]")
        print(f"Q: {question[:80]}...")
        print(f"Q Entities: nums={q_entities['nums'][:5]}, names={q_entities['names'][:5]}")
        print(f"Output: {output_text[:100]}...")
        print(f"Pred: {pred_answer} | GT: {gt_answer} | {'CORRECT' if is_correct else 'wrong'}")
        print(f"Matches: nums={num_matches}/{len(q_entities['nums'])}, "
              f"names={name_matches}/{len(q_entities['names'])}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 15 EVALUATION SUMMARY")
    print("=" * 70)

    num_pct = 100 * matched_nums / max(total_nums, 1)
    name_pct = 100 * matched_names / max(total_names, 1)
    total_entities = total_nums + total_names
    total_matches = matched_nums + matched_names
    overall_pct = 100 * total_matches / max(total_entities, 1)
    unique_pct = 100 * len(unique_outputs) / args.num_samples
    accuracy = 100 * correct_answers / args.num_samples

    print(f"Answer Accuracy: {correct_answers}/{args.num_samples} ({accuracy:.1f}%)")
    print(f"Numbers: {matched_nums}/{total_nums} matched ({num_pct:.1f}%)")
    print(f"Names: {matched_names}/{total_names} matched ({name_pct:.1f}%)")
    print(f"\nOVERALL ENTITY TRANSFER: {total_matches}/{total_entities} ({overall_pct:.1f}%)")
    print(f"\nOUTPUT DIVERSITY: {len(unique_outputs)}/{args.num_samples} unique ({unique_pct:.1f}%)")
    print("")

    # Diagnosis
    if unique_pct < 50:
        print("=" * 70)
        print("LOW DIVERSITY: Outputs are similar/identical.")
        print("VQ may be collapsing to few codes.")
        print("Consider: Check perplexity, increase codebook diversity")
        print("=" * 70)
    elif overall_pct < 10:
        print("=" * 70)
        print("VERY LOW ENTITY TRANSFER (<10%).")
        print("VQ codes may not capture entity information.")
        print("Consider: Increase codebook size, more training")
        print("=" * 70)
    elif overall_pct < 30:
        print("=" * 70)
        print("PARTIAL SUCCESS: Entity transfer 10-30%.")
        print("VQ is helping but not fully solving the problem.")
        print("Consider: More training, adjust VQ weight")
        print("=" * 70)
    elif overall_pct >= 30:
        print("=" * 70)
        print("SUCCESS! Entity transfer >= 30%!")
        print("VQ-Telepathy is working!")
        print("Next: Tune hyperparameters, increase training")
        print("=" * 70)

    # Save results
    output_path = os.path.join(args.output_dir, "eval_v15_results.json")
    summary = {
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'accuracy': accuracy,
        'correct_answers': correct_answers,
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
