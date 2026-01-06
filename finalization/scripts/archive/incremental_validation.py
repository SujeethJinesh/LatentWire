#!/usr/bin/env python3
"""
Incremental validation script to identify where the pipeline breaks.
We know embeddings work (82% F1), so let's add components one by one.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from latentwire.models import ByteInterlinguaEncoder, Adapter, LMWrapper
from latentwire.eval import evaluate_on_squad
from latentwire.data import load_squad_dataset
import argparse

def test_step_1_raw_embeddings(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", samples=200):
    """Step 1: Validate raw embeddings still work (baseline)"""
    print("\n" + "="*60)
    print("STEP 1: Raw Embeddings (Known Working Baseline)")
    print("="*60)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Get test data
    dataset = load_squad_dataset("validation", samples)

    # Test with raw embeddings
    results = []
    for item in dataset[:10]:  # Quick test
        # Tokenize
        text = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Get embeddings directly
        with torch.no_grad():
            embeds = model.get_input_embeddings()(inputs.input_ids)
            outputs = model(inputs_embeds=embeds, max_new_tokens=12)

        # Decode
        generated = tokenizer.decode(outputs.logits.argmax(-1)[0])
        results.append({
            "gold": item["answer"],
            "pred": generated,
            "match": item["answer"].lower() in generated.lower()
        })

    accuracy = sum(r["match"] for r in results) / len(results)
    print(f"✓ Raw embeddings accuracy: {accuracy:.1%}")
    print(f"  Expected: ~80-82%")
    return accuracy > 0.7  # Should be ~0.82

def test_step_2_identity_adapter(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", samples=200):
    """Step 2: Add identity adapter (should preserve performance)"""
    print("\n" + "="*60)
    print("STEP 2: Identity Adapter (Linear projection)")
    print("="*60)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create identity adapter
    embed_dim = model.config.hidden_size
    adapter = nn.Linear(embed_dim, embed_dim)

    # Initialize as identity
    with torch.no_grad():
        adapter.weight.data = torch.eye(embed_dim)
        adapter.bias.data.zero_()

    adapter = adapter.to(model.device)

    # Test
    dataset = load_squad_dataset("validation", samples)
    results = []

    for item in dataset[:10]:
        text = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            # Get embeddings
            embeds = model.get_input_embeddings()(inputs.input_ids)
            # Pass through adapter
            adapted_embeds = adapter(embeds)
            # Generate
            outputs = model(inputs_embeds=adapted_embeds, max_new_tokens=12)

        generated = tokenizer.decode(outputs.logits.argmax(-1)[0])
        results.append({
            "gold": item["answer"],
            "pred": generated,
            "match": item["answer"].lower() in generated.lower()
        })

    accuracy = sum(r["match"] for r in results) / len(results)
    print(f"✓ Identity adapter accuracy: {accuracy:.1%}")
    print(f"  Expected: ~80-82% (same as raw)")
    return accuracy > 0.7

def test_step_3_trained_adapter(checkpoint_path, samples=200):
    """Step 3: Test with trained adapter only (no encoder)"""
    print("\n" + "="*60)
    print("STEP 3: Trained Adapter (from checkpoint)")
    print("="*60)

    # Load checkpoint
    ckpt = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")
    config = json.load(open(f"{checkpoint_path}/config.json"))

    # Load model
    model_id = config["llama_id"]
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load adapter
    embed_dim = model.config.hidden_size
    adapter = Adapter(
        d_in=config["d_z"],
        d_out=embed_dim,
        hidden_mult=config.get("adapter_hidden_mult", 2),
        dropout=config.get("adapter_dropout", 0.0)
    ).to(model.device)

    adapter.load_state_dict(ckpt["adapter_llama"])
    adapter.eval()

    # Test with random latents (simulate encoder output)
    dataset = load_squad_dataset("validation", samples)
    results = []

    for item in dataset[:10]:
        with torch.no_grad():
            # Create random latents of correct size
            batch_size = 1
            latent_len = config["latent_len"]
            d_z = config["d_z"]

            # Sample from similar distribution as embeddings
            z = torch.randn(batch_size, latent_len, d_z).to(model.device) * 0.02

            # Pass through adapter
            adapted = adapter(z)

            # Generate
            outputs = model(inputs_embeds=adapted, max_new_tokens=12)

        generated = tokenizer.decode(outputs.logits.argmax(-1)[0])
        print(f"  Generated: {generated[:50]}...")

    print(f"✓ Adapter processes latents without crashing")
    return True

def test_step_4_encoder_alone(checkpoint_path, samples=200):
    """Step 4: Test encoder output statistics"""
    print("\n" + "="*60)
    print("STEP 4: Encoder Output Analysis")
    print("="*60)

    # Load checkpoint
    ckpt = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")
    config = json.load(open(f"{checkpoint_path}/config.json"))

    # Load encoder
    encoder = ByteEncoder(
        d_model=config["d_z"],
        max_seq_len=config.get("byte_max", 512),
        latent_len=config["latent_len"]
    )

    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Test encoder outputs
    dataset = load_squad_dataset("validation", samples)
    latent_stats = []

    for item in dataset[:50]:
        text = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer: "
        text_bytes = text.encode('utf-8')[:512]

        with torch.no_grad():
            # Encode
            z = encoder(torch.tensor([list(text_bytes)]))

            latent_stats.append({
                "mean": z.mean().item(),
                "std": z.std().item(),
                "norm": z.norm(dim=-1).mean().item(),
                "max": z.abs().max().item()
            })

    # Aggregate stats
    import numpy as np
    for key in ["mean", "std", "norm", "max"]:
        values = [s[key] for s in latent_stats]
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n  Compare to embedding stats (should be similar):")
    print("  Typical embedding norm: ~1.0")
    print("  Typical embedding std: ~0.02")

    avg_norm = np.mean([s["norm"] for s in latent_stats])
    return 0.5 < avg_norm < 2.0  # Reasonable range

def test_step_5_full_pipeline_debug(checkpoint_path, samples=10):
    """Step 5: Full pipeline with detailed debugging"""
    print("\n" + "="*60)
    print("STEP 5: Full Pipeline Debugging")
    print("="*60)

    # Load everything
    ckpt = torch.load(f"{checkpoint_path}/model.pt", map_location="cpu")
    config = json.load(open(f"{checkpoint_path}/config.json"))

    model_id = config["llama_id"]
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load encoder
    encoder = ByteEncoder(
        d_model=config["d_z"],
        max_seq_len=config.get("byte_max", 512),
        latent_len=config["latent_len"]
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Load adapter
    adapter = Adapter(
        d_in=config["d_z"],
        d_out=model.config.hidden_size,
        hidden_mult=config.get("adapter_hidden_mult", 2),
        dropout=0.0  # No dropout in eval
    ).to(model.device)
    adapter.load_state_dict(ckpt["adapter_llama"])
    adapter.eval()

    # Test one example with detailed logging
    dataset = load_squad_dataset("validation", samples)
    item = dataset[0]

    text = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer: "
    print(f"\n  Input text: {text[:100]}...")
    print(f"  Gold answer: {item['answer']}")

    # Encode
    text_bytes = text.encode('utf-8')[:512]
    with torch.no_grad():
        z = encoder(torch.tensor([list(text_bytes)]))
        print(f"\n  Encoder output shape: {z.shape}")
        print(f"  Encoder output stats: mean={z.mean():.4f}, std={z.std():.4f}, norm={z.norm(dim=-1).mean():.4f}")

        # Adapt
        embeds = adapter(z.to(model.device))
        print(f"\n  Adapter output shape: {embeds.shape}")
        print(f"  Adapter output stats: mean={embeds.mean():.4f}, std={embeds.std():.4f}, norm={embeds.norm(dim=-1).mean():.4f}")

        # Compare to real embeddings
        real_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        real_embeds = model.get_input_embeddings()(real_inputs.input_ids.to(model.device))
        print(f"\n  Real embedding stats: mean={real_embeds.mean():.4f}, std={real_embeds.std():.4f}, norm={real_embeds.norm(dim=-1).mean():.4f}")

        # Generate
        outputs = model.generate(
            inputs_embeds=embeds,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(outputs[0])
    print(f"\n  Generated: {generated}")

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--step", type=int, help="Which step to run (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all or args.step == 1:
        success = test_step_1_raw_embeddings()
        if not success:
            print("❌ Step 1 failed - raw embeddings not working")
            return

    if args.all or args.step == 2:
        success = test_step_2_identity_adapter()
        if not success:
            print("❌ Step 2 failed - identity adapter breaks embeddings")
            return

    if args.checkpoint and (args.all or args.step == 3):
        success = test_step_3_trained_adapter(args.checkpoint)
        if not success:
            print("❌ Step 3 failed - trained adapter issues")
            return

    if args.checkpoint and (args.all or args.step == 4):
        success = test_step_4_encoder_alone(args.checkpoint)
        if not success:
            print("❌ Step 4 failed - encoder output issues")
            return

    if args.checkpoint and (args.all or args.step == 5):
        success = test_step_5_full_pipeline_debug(args.checkpoint)
        if not success:
            print("❌ Step 5 failed - full pipeline issues")
            return

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()