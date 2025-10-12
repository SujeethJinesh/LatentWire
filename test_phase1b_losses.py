#!/usr/bin/env python3
"""
Test K-token CE and Prefix KD losses on text baseline.
This verifies the loss functions themselves work correctly.
"""

import torch
from latentwire.models import LMWrapper, LMConfig
from latentwire.losses import k_token_ce_from_prefix, kd_first_k_prefix_vs_text
from datasets import load_dataset

def test_losses_on_text():
    """Test that losses work on text baseline (should be low)."""

    print("\n" + "="*60)
    print("TESTING PHASE 1B LOSS FUNCTIONS ON TEXT BASELINE")
    print("="*60)
    print("\nGoal: Verify K-token CE and Prefix KD give LOW loss on text")
    print("If losses are high even for text, functions are broken!\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model on {device} with dtype {dtype}...")
    lm_config = LMConfig(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        device=device,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    llm = LMWrapper(lm_config)
    tokenizer = llm.tokenizer

    # Load one SQuAD example
    print("Loading SQuAD example...")
    ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
    example = next(iter(ds))

    context = example['context']
    question = example['question']
    answer = example['answers']['text'][0]

    print(f"\nExample:")
    print(f"  Question: {question}")
    print(f"  Answer: {answer}")

    # Create prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False, return_tensors='pt').to(device)
    anchor_ids = tokenizer.encode("Answer:", add_special_tokens=False, return_tensors='pt').to(device)

    print(f"\n  Input tokens: {input_ids.shape[1]}")
    print(f"  Answer tokens: {answer_ids.shape[1]}")

    # Get embeddings for the prompt (text baseline)
    with torch.no_grad():
        embed_layer = llm.model.get_input_embeddings()
        prefix_embeds = embed_layer(input_ids)

    print(f"\n  Prefix embeds: {prefix_embeds.shape}")

    # Test K-token CE loss
    print("\n" + "-"*60)
    print("TEST 1: K-token CE Loss (k_token_ce_from_prefix)")
    print("-"*60)

    for K in [2, 4, 8]:
        try:
            loss_kce = k_token_ce_from_prefix(
                llm,
                prefix_embeds=prefix_embeds,
                gold_ids=answer_ids,
                K=K,
                anchor_ids=anchor_ids,
                append_bos_after_prefix=True
            )
            print(f"  K={K}: loss_kce = {loss_kce.item():.4f}")
        except Exception as e:
            print(f"  K={K}: ERROR - {e}")

    # Test Prefix KD loss
    print("\n" + "-"*60)
    print("TEST 2: Prefix KD Loss (kd_first_k_prefix_vs_text)")
    print("-"*60)

    for K in [2, 4, 8]:
        for tau in [0.5, 1.0, 2.0]:
            try:
                loss_kd = kd_first_k_prefix_vs_text(
                    student_llm=llm,
                    teacher_llm=llm,
                    prefix_embeds=prefix_embeds,
                    scaffold_ids=input_ids,
                    gold_ids=answer_ids,
                    K=K,
                    tau=tau,
                    anchor_ids=anchor_ids,
                    append_bos_after_prefix=True
                )
                print(f"  K={K}, tau={tau}: loss_kd = {loss_kd.item():.4f}")
            except Exception as e:
                print(f"  K={K}, tau={tau}: ERROR - {e}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("\nFor text baseline (perfect input), we expect:")
    print("  - K-token CE: ~1-3 (predicting next token is hard even for text)")
    print("  - Prefix KD: ~0.01-0.1 (student=teacher, should match closely)")
    print("\nIf losses are much higher (>5), functions may be broken!")
    print("If losses are in expected range, Phase 1b issue is weight tuning.")
    print("\n")

if __name__ == "__main__":
    test_losses_on_text()
