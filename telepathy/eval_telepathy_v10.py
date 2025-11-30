#!/usr/bin/env python
# telepathy/eval_telepathy_v10.py
"""
Phase 10 Evaluation: Does Mistral "Read Back" the Question?

Success Criteria:
- Mistral output STARTS with question content (Janet, ducks, 16, etc)
- Then provides the answer

This proves the soft tokens encode question info in Mistral-readable format.
"""
import torch
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from latent_bridge_v9 import LatentBridgeV9
import argparse


def extract_entities(text):
    """Extract key entities (names, numbers) from text for comparison."""
    # Find all numbers
    numbers = set(re.findall(r'\b\d+\b', text))
    # Find capitalized words (names)
    names = set(re.findall(r'\b[A-Z][a-z]+\b', text))
    # Find key nouns (simplified)
    nouns = set(re.findall(r'\b(ducks?|eggs?|chickens?|students?|apples?|friends?|farmers?|market)\b', text.lower()))
    return numbers, names, nouns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    SOURCE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    TARGET_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    print("=" * 70)
    print("Phase 10 Evaluation: Auto-Encoder Test")
    print("=" * 70)
    print("Success = Mistral 'reads back' the question content from vectors")
    print("Look for: Entity names, numbers, key nouns from original question")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Load bridge
    class BridgeArgs:
        stats_path = args.stats_path
        soft_tokens = args.soft_tokens
        heads = args.heads
        depth = args.depth

    bridge = LatentBridgeV9(
        BridgeArgs(),
        src_model.config.hidden_size,
        tgt_model.config.hidden_size,
        src_vocab_size=src_model.config.vocab_size
    )
    bridge.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    bridge.to(DEVICE).bfloat16().eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Output scale: {bridge.output_scale.item():.4f}")

    # Diagnostic: Check embedding scale comparison
    with torch.no_grad():
        tgt_emb_weight = tgt_model.get_input_embeddings().weight
        tgt_rms = tgt_emb_weight.float().pow(2).mean(dim=1).sqrt().median().item()
        print(f"Target embedding RMS: {tgt_rms:.4f}")
        print(f"Soft tokens will have ~{bridge.output_scale.item()/tgt_rms:.2f}x target scale")

    # Load test data
    ds = load_dataset("gsm8k", "main", split="test")

    # Sample indices spread across dataset
    total = len(ds)
    indices = [int(i * total / args.num_samples) for i in range(args.num_samples)]

    results = []
    entity_matches = {"numbers": 0, "names": 0, "nouns": 0}
    total_entities = {"numbers": 0, "names": 0, "nouns": 0}

    print(f"\nEvaluating {args.num_samples} samples...")
    print("-" * 70)

    for idx in indices:
        question = ds[idx]['question']
        answer = ds[idx]['answer']

        # Extract entities from original question
        q_numbers, q_names, q_nouns = extract_entities(question)
        total_entities["numbers"] += len(q_numbers)
        total_entities["names"] += len(q_names)
        total_entities["nouns"] += len(q_nouns)

        # Source forward (Llama reads question)
        src_input = f"Question: {question}\nAnswer:"
        src_enc = src_tok(src_input, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer].bfloat16()
            soft_tokens, _ = bridge(src_h, src_enc.attention_mask)

            # FIX: Add BOS token to kick off generation
            # Training always had BOS after soft tokens, so eval needs it too
            bos_emb = tgt_model.get_input_embeddings()(
                torch.tensor([[tgt_tok.bos_token_id]], device=DEVICE)
            ).bfloat16()
            inputs_embeds = torch.cat([soft_tokens, bos_emb], dim=1)

            # FIX: Add attention mask - required for proper attention
            attention_mask = torch.ones(
                1, inputs_embeds.shape[1], device=DEVICE, dtype=torch.long
            )

            # Generate from soft tokens + BOS
            # Mistral should reconstruct question content
            out_ids = tgt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        output = tgt_tok.decode(out_ids[0], skip_special_tokens=True)

        # Debug: Show raw output for first sample
        if idx == indices[0]:
            raw_output = tgt_tok.decode(out_ids[0], skip_special_tokens=False)
            print(f"\n[DEBUG] First sample raw output: {repr(raw_output[:200])}")
            print(f"[DEBUG] Generated {len(out_ids[0])} tokens, soft_tokens shape: {soft_tokens.shape}")
            print(f"[DEBUG] Soft token stats - min: {soft_tokens.min():.4f}, max: {soft_tokens.max():.4f}, mean: {soft_tokens.mean():.4f}")

        # Extract entities from output
        o_numbers, o_names, o_nouns = extract_entities(output)

        # Count matches
        num_match = len(q_numbers & o_numbers)
        name_match = len(q_names & o_names)
        noun_match = len(q_nouns & o_nouns)

        entity_matches["numbers"] += num_match
        entity_matches["names"] += name_match
        entity_matches["nouns"] += noun_match

        result = {
            "index": idx,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "output": output[:200] + "..." if len(output) > 200 else output,
            "q_entities": {"numbers": list(q_numbers), "names": list(q_names), "nouns": list(q_nouns)},
            "o_entities": {"numbers": list(o_numbers), "names": list(o_names), "nouns": list(o_nouns)},
            "matches": {"numbers": num_match, "names": name_match, "nouns": noun_match}
        }
        results.append(result)

        # Print sample
        print(f"\n[Sample {idx}]")
        print(f"Q Entities: nums={list(q_numbers)[:3]}, names={list(q_names)[:3]}, nouns={list(q_nouns)[:3]}")
        print(f"Output: {output[:150]}...")
        print(f"O Entities: nums={list(o_numbers)[:3]}, names={list(o_names)[:3]}, nouns={list(o_nouns)[:3]}")
        matches_str = f"Matches: nums={num_match}/{len(q_numbers)}, names={name_match}/{len(q_names)}, nouns={noun_match}/{len(q_nouns)}"
        print(matches_str)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 10 EVALUATION SUMMARY")
    print("=" * 70)

    for key in ["numbers", "names", "nouns"]:
        total = total_entities[key]
        matched = entity_matches[key]
        rate = (matched / total * 100) if total > 0 else 0
        print(f"{key.capitalize()}: {matched}/{total} matched ({rate:.1f}%)")

    overall_total = sum(total_entities.values())
    overall_matched = sum(entity_matches.values())
    overall_rate = (overall_matched / overall_total * 100) if overall_total > 0 else 0
    print(f"\nOVERALL ENTITY TRANSFER: {overall_matched}/{overall_total} ({overall_rate:.1f}%)")

    print("\n" + "=" * 70)
    if overall_rate > 30:
        print("PROGRESS! Entities are transferring through soft tokens.")
        print("The Auto-Encoder approach is working.")
    elif overall_rate > 10:
        print("PARTIAL SUCCESS. Some entity transfer detected.")
        print("May need more training or larger soft token count.")
    else:
        print("STILL FAILING. Entities not transferring.")
        print("Check if loss decreased during training.")
    print("=" * 70)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "config": {
                "checkpoint": args.checkpoint,
                "soft_tokens": args.soft_tokens,
                "num_samples": args.num_samples
            },
            "entity_matches": entity_matches,
            "total_entities": total_entities,
            "overall_rate": overall_rate,
            "results": results
        }

        with open(output_dir / "eval_v10_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_dir / 'eval_v10_results.json'}")


if __name__ == "__main__":
    main()
