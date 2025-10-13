#!/usr/bin/env python3
"""
Test script for Anchor-Guided Cross-Model Interlingua architecture.

Quick validation to ensure:
1. Components can be instantiated
2. Forward pass works
3. Loss computation succeeds
4. Basic training step completes
5. Generation produces non-collapsed output (NOT all identical like ByteEncoder)

Usage (via wrapper script):
    # Quick smoke test (5 min)
    git pull && rm -rf runs && PYTHONPATH=. SAMPLES=10 STEPS=5 bash scripts/test_new_interlingua.sh

    # Realistic test (30 min)
    git pull && rm -rf runs && PYTHONPATH=. SAMPLES=100 STEPS=50 bash scripts/test_new_interlingua.sh

    # Full test with Qwen (2 hours)
    git pull && rm -rf runs && PYTHONPATH=. SAMPLES=1000 STEPS=500 TEST_QWEN=yes bash scripts/test_new_interlingua.sh

Direct usage:
    python scripts/test_new_interlingua.py --samples 100 --steps 50
"""

import argparse
import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import sentence_transformers, install if missing
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

from latentwire.data import load_squad_subset


# ============================================================================
# NEW ARCHITECTURE COMPONENTS
# ============================================================================

class AlignmentTransformer(nn.Module):
    """Projects token embeddings to shared interlingua space, guided by semantic anchor.

    Key features:
    - Per-model input projections (handles different d_model)
    - Cross-attention to frozen semantic anchor
    - Transformer encoder for refinement
    - Mean pooling to single vector
    """

    def __init__(
        self,
        d_model_llama: int = 4096,
        d_model_qwen: int = 2048,
        d_sem: int = 384,
        d_inter: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_inter = d_inter

        # Per-model input projections
        self.proj_llama = nn.Linear(d_model_llama, d_inter)
        self.proj_qwen = nn.Linear(d_model_qwen, d_inter)
        self.proj_sem = nn.Linear(d_sem, d_inter)

        # Cross-attention to semantic anchor
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_inter, num_heads=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_inter) for _ in range(n_layers)])

        # Refinement transformer
        self.refine = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_inter,
                nhead=n_heads,
                dim_feedforward=d_inter * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )

        self.final_norm = nn.LayerNorm(d_inter)

    def forward(
        self,
        token_embeds: torch.Tensor,
        sem_anchor: torch.Tensor,
        model_type: str = 'llama',
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_embeds: [B, T, d_model] - frozen LLM token embeddings
            sem_anchor: [B, d_sem] - frozen semantic encoder output
            model_type: 'llama' or 'qwen'
            attention_mask: [B, T] - attention mask (1=valid, 0=padding)

        Returns:
            z: [B, d_inter] - interlingua representation
        """
        # Project to shared dimension
        if model_type == 'llama':
            x = self.proj_llama(token_embeds)  # [B, T, d_inter]
        elif model_type == 'qwen':
            x = self.proj_qwen(token_embeds)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Prepare semantic anchor
        sem = self.proj_sem(sem_anchor).unsqueeze(1)  # [B, 1, d_inter]

        # Cross-attend to semantic anchor
        for attn_layer, norm_layer in zip(self.cross_attn_layers, self.cross_attn_norms):
            attn_out, _ = attn_layer(
                query=x,
                key=sem,
                value=sem,
                need_weights=False,
            )
            x = norm_layer(x + attn_out)  # Residual connection

        # Refine with self-attention
        if attention_mask is not None:
            # Convert mask to correct format: True = ignore, False = attend
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.refine(x, src_key_padding_mask=key_padding_mask)  # [B, T, d_inter]

        # Mean pool (excluding padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            x_masked = x * mask_expanded
            z = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            z = x.mean(dim=1)  # [B, d_inter]

        z = self.final_norm(z)
        return z


class InterlinguaAdapter(nn.Module):
    """Expands interlingua vector to M soft tokens in LLM embedding space.

    Key features:
    - Expand single vector to sequence of soft tokens
    - Learned queries for position awareness
    - Project to LLM's d_model
    - Calibration scale parameter
    """

    def __init__(
        self,
        d_inter: int = 512,
        d_model: int = 4096,
        num_slots: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_inter = d_inter
        self.d_model = d_model

        # Learned slot queries
        self.queries = nn.Parameter(torch.randn(num_slots, d_inter) * 0.02)

        # Expand single vector to sequence
        self.expand = nn.Sequential(
            nn.Linear(d_inter, d_inter * num_slots),
            nn.LayerNorm(d_inter * num_slots),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Project to LLM space
        self.proj = nn.Sequential(
            nn.Linear(d_inter, d_model),
            nn.LayerNorm(d_model),
        )

        # Calibration scale
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, z_inter: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_inter: [B, d_inter] - interlingua representation

        Returns:
            embeds: [B, M, d_model] - soft token embeddings for LLM
        """
        B = z_inter.size(0)

        # Expand to sequence
        expanded = self.expand(z_inter)  # [B, d_inter * M]
        seq = expanded.view(B, self.num_slots, self.d_inter)  # [B, M, d_inter]

        # Add learned positional queries
        seq = seq + self.queries.unsqueeze(0)  # [B, M, d_inter]

        # Project to LLM embedding space
        embeds = self.proj(seq)  # [B, M, d_model]
        embeds = embeds * self.scale

        return embeds


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def k_token_ce_simple(
    model,
    tokenizer,
    prefix_embeds: torch.Tensor,
    answer_text: str,
    K: int = 4,
    anchor_text: str = "Answer: ",
) -> torch.Tensor:
    """Simplified K-token CE for testing (no deep prefix, no latent adapters)."""
    device = prefix_embeds.device

    # Tokenize answer
    answer_ids = tokenizer(answer_text, return_tensors='pt').input_ids.to(device)

    # Tokenize anchor
    anchor_ids = tokenizer(anchor_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    total_loss = 0.0
    steps = 0

    for t in range(min(K, answer_ids.size(1))):
        # Build input: [prefix, anchor, answer[:t]]
        if t == 0:
            # Just prefix + anchor
            inputs_embeds = torch.cat([
                prefix_embeds,
                model.get_input_embeddings()(anchor_ids),
            ], dim=1)
        else:
            # Prefix + anchor + partial answer
            prev_answer_ids = answer_ids[:, :t]
            inputs_embeds = torch.cat([
                prefix_embeds,
                model.get_input_embeddings()(anchor_ids),
                model.get_input_embeddings()(prev_answer_ids),
            ], dim=1)

        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits[:, -1, :]  # Last position logits

        # Target is next token
        target = answer_ids[:, t]
        loss = F.cross_entropy(logits, target, reduction='mean')
        total_loss += loss
        steps += 1

    return total_loss / max(steps, 1)


def generate_from_prefix(
    model,
    tokenizer,
    prefix_embeds: torch.Tensor,
    anchor_text: str = "Answer: ",
    max_new_tokens: int = 12,
    temperature: float = 0.0,
) -> str:
    """Generate text from soft prefix embeddings."""
    device = prefix_embeds.device

    # Tokenize anchor
    anchor_ids = tokenizer(anchor_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    anchor_embeds = model.get_input_embeddings()(anchor_ids)

    # Initial input: prefix + anchor
    inputs_embeds = torch.cat([prefix_embeds, anchor_embeds], dim=1)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode (skip the prefix tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove anchor text if present
    if anchor_text and generated_text.startswith(anchor_text):
        generated_text = generated_text[len(anchor_text):]

    return generated_text.strip()


# ============================================================================
# TEST SCRIPT
# ============================================================================

def test_architecture(args):
    """Main test function."""

    print("=" * 80)
    print("TESTING: Anchor-Guided Cross-Model Interlingua Architecture")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ========================================================================
    # 1. Load frozen models
    # ========================================================================

    print("\n[1/7] Loading frozen models...")

    # Semantic encoder (frozen)
    print("  - SentenceTransformer...")
    sem_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sem_encoder.eval()
    for p in sem_encoder.parameters():
        p.requires_grad = False
    d_sem = 384

    # Llama (frozen)
    print("  - Llama-3.1-8B...")
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_id, use_fast=True)
    llama_tokenizer.padding_side = 'left'
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    llama_model.eval()
    for p in llama_model.parameters():
        p.requires_grad = False
    d_model_llama = llama_model.config.hidden_size

    # Only load Qwen if requested
    if args.test_qwen:
        print("  - Qwen2.5-7B...")
        qwen_id = "Qwen/Qwen2.5-7B-Instruct"
        qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_id, use_fast=True)
        qwen_tokenizer.padding_side = 'left'
        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )
        qwen_model.eval()
        for p in qwen_model.parameters():
            p.requires_grad = False
        d_model_qwen = qwen_model.config.hidden_size
    else:
        qwen_model = None
        qwen_tokenizer = None
        d_model_qwen = 2048  # default

    print("  ✓ All models loaded")

    # ========================================================================
    # 2. Instantiate new architecture components
    # ========================================================================

    print("\n[2/7] Instantiating new architecture components...")

    # Determine dtype from frozen models (bfloat16 on CUDA, float32 on CPU)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    alignment_tf = AlignmentTransformer(
        d_model_llama=d_model_llama,
        d_model_qwen=d_model_qwen,
        d_sem=d_sem,
        d_inter=args.d_inter,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    ).to(device=device, dtype=model_dtype)

    adapter_llama = InterlinguaAdapter(
        d_inter=args.d_inter,
        d_model=d_model_llama,
        num_slots=args.num_slots,
        dropout=0.1,
    ).to(device=device, dtype=model_dtype)

    if qwen_model is not None:
        adapter_qwen = InterlinguaAdapter(
            d_inter=args.d_inter,
            d_model=d_model_qwen,
            num_slots=args.num_slots,
            dropout=0.1,
        ).to(device=device, dtype=model_dtype)
    else:
        adapter_qwen = None

    # Count parameters
    total_params = sum(p.numel() for p in alignment_tf.parameters())
    total_params += sum(p.numel() for p in adapter_llama.parameters())
    if adapter_qwen is not None:
        total_params += sum(p.numel() for p in adapter_qwen.parameters())

    print(f"  ✓ AlignmentTransformer: {sum(p.numel() for p in alignment_tf.parameters()):,} params")
    print(f"  ✓ InterlinguaAdapter (Llama): {sum(p.numel() for p in adapter_llama.parameters()):,} params")
    if adapter_qwen is not None:
        print(f"  ✓ InterlinguaAdapter (Qwen): {sum(p.numel() for p in adapter_qwen.parameters()):,} params")
    print(f"  Total trainable: {total_params:,} params ({total_params/1e6:.1f}M)")

    # ========================================================================
    # 3. Load test data
    # ========================================================================

    print(f"\n[3/7] Loading test data (SQuAD, n={args.samples})...")

    examples = load_squad_subset(split='validation', samples=args.samples)
    print(f"  ✓ Loaded {len(examples)} examples")
    print(f"  Example: {examples[0]['source'][:100]}...")

    # ========================================================================
    # 4. Test forward pass (NO training)
    # ========================================================================

    print("\n[4/7] Testing forward pass (single example)...")

    example = examples[0]
    text = example['source']
    answer = example['answer']

    # Encode with semantic encoder
    with torch.no_grad():
        z_sem = torch.tensor(
            sem_encoder.encode([text], convert_to_tensor=False, show_progress_bar=False),
            dtype=model_dtype,
            device=device,
        )

    # Tokenize and embed with Llama
    llama_tokens = llama_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    llama_input_ids = llama_tokens.input_ids.to(device)
    llama_attn_mask = llama_tokens.attention_mask.to(device)

    with torch.no_grad():
        llama_embeds = llama_model.get_input_embeddings()(llama_input_ids)

    # Forward through alignment transformer
    z_llama = alignment_tf(
        token_embeds=llama_embeds,
        sem_anchor=z_sem,
        model_type='llama',
        attention_mask=llama_attn_mask,
    )

    # Forward through adapter
    prefix_embeds_llama = adapter_llama(z_llama)

    print(f"  ✓ z_sem shape: {z_sem.shape}")
    print(f"  ✓ llama_embeds shape: {llama_embeds.shape}")
    print(f"  ✓ z_llama shape: {z_llama.shape}")
    print(f"  ✓ prefix_embeds_llama shape: {prefix_embeds_llama.shape}")
    print(f"  ✓ Expected: [1, {args.num_slots}, {d_model_llama}]")

    # Test generation BEFORE training (should be random)
    print("\n  Testing generation (before training)...")
    pred_before = generate_from_prefix(
        llama_model,
        llama_tokenizer,
        prefix_embeds_llama,
        anchor_text="Answer: ",
        max_new_tokens=12,
    )
    print(f"  Gold answer: {answer}")
    print(f"  Pred (untrained): {pred_before}")

    if args.test_qwen and qwen_model is not None:
        # Test with Qwen
        qwen_tokens = qwen_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        qwen_input_ids = qwen_tokens.input_ids.to(device)
        qwen_attn_mask = qwen_tokens.attention_mask.to(device)

        with torch.no_grad():
            qwen_embeds = qwen_model.get_input_embeddings()(qwen_input_ids)

        z_qwen = alignment_tf(
            token_embeds=qwen_embeds,
            sem_anchor=z_sem,
            model_type='qwen',
            attention_mask=qwen_attn_mask,
        )

        prefix_embeds_qwen = adapter_qwen(z_qwen)

        print(f"  ✓ z_qwen shape: {z_qwen.shape}")
        print(f"  ✓ prefix_embeds_qwen shape: {prefix_embeds_qwen.shape}")

        # Test alignment (should be random before training)
        alignment_dist = F.mse_loss(z_llama, z_qwen).item()
        print(f"  Alignment distance (untrained): {alignment_dist:.4f}")

    # ========================================================================
    # 5. Test loss computation
    # ========================================================================

    print("\n[5/7] Testing loss computation...")

    # Generation loss
    loss_gen = k_token_ce_simple(
        llama_model,
        llama_tokenizer,
        prefix_embeds_llama,
        answer,
        K=4,
        anchor_text="Answer: ",
    )
    print(f"  ✓ Generation loss (K=4): {loss_gen.item():.4f}")

    if args.test_qwen and qwen_model is not None:
        # Alignment loss
        loss_align = F.mse_loss(z_llama, z_qwen)
        print(f"  ✓ Alignment loss: {loss_align.item():.4f}")

        # Semantic anchor loss
        z_sem_proj = alignment_tf.proj_sem(z_sem)
        loss_sem = F.mse_loss(z_llama, z_sem_proj) + F.mse_loss(z_qwen, z_sem_proj)
        print(f"  ✓ Semantic anchor loss: {loss_sem.item():.4f}")

        # Total loss
        total_loss = loss_gen + 0.5 * loss_align + 0.1 * loss_sem
        print(f"  ✓ Total loss: {total_loss.item():.4f}")
    else:
        total_loss = loss_gen
        print(f"  ✓ Total loss: {total_loss.item():.4f}")

    # ========================================================================
    # 6. Run training steps (if requested)
    # ========================================================================

    if args.steps > 0:
        print(f"\n[6/7] Running {args.steps} training steps on {args.samples} examples...")

        # Create optimizer
        params = list(alignment_tf.parameters()) + list(adapter_llama.parameters())
        if adapter_qwen is not None:
            params += list(adapter_qwen.parameters())

        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

        # Training loop
        alignment_tf.train()
        adapter_llama.train()
        if adapter_qwen is not None:
            adapter_qwen.train()

        for step in range(args.steps):
            # Sample a batch
            batch_indices = torch.randint(0, len(examples), (args.batch_size,))
            batch = [examples[i] for i in batch_indices]

            optimizer.zero_grad()

            batch_loss = 0.0
            batch_gen_loss = 0.0
            batch_align_loss = 0.0
            batch_sem_loss = 0.0

            for ex in batch:
                text = ex['source']
                answer = ex['answer']

                # Encode with semantic encoder
                with torch.no_grad():
                    z_sem = torch.tensor(
                        sem_encoder.encode([text], convert_to_tensor=False, show_progress_bar=False),
                        dtype=model_dtype,
                        device=device,
                    )

                # Llama
                llama_tokens = llama_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                llama_input_ids = llama_tokens.input_ids.to(device)
                llama_attn_mask = llama_tokens.attention_mask.to(device)

                with torch.no_grad():
                    llama_embeds = llama_model.get_input_embeddings()(llama_input_ids)

                z_llama = alignment_tf(llama_embeds, z_sem, 'llama', llama_attn_mask)
                prefix_llama = adapter_llama(z_llama)

                # Generation loss (Llama)
                loss_gen_llama = k_token_ce_simple(
                    llama_model, llama_tokenizer, prefix_llama, answer, K=4, anchor_text="Answer: "
                )

                # Qwen (if enabled)
                if args.test_qwen and qwen_model is not None:
                    qwen_tokens = qwen_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    qwen_input_ids = qwen_tokens.input_ids.to(device)
                    qwen_attn_mask = qwen_tokens.attention_mask.to(device)

                    with torch.no_grad():
                        qwen_embeds = qwen_model.get_input_embeddings()(qwen_input_ids)

                    z_qwen = alignment_tf(qwen_embeds, z_sem, 'qwen', qwen_attn_mask)
                    prefix_qwen = adapter_qwen(z_qwen)

                    # Generation loss (Qwen)
                    loss_gen_qwen = k_token_ce_simple(
                        qwen_model, qwen_tokenizer, prefix_qwen, answer, K=4, anchor_text="Answer: "
                    )

                    # Alignment loss
                    loss_align = F.mse_loss(z_llama, z_qwen)

                    # Semantic anchor loss
                    z_sem_proj = alignment_tf.proj_sem(z_sem)
                    loss_sem = F.mse_loss(z_llama, z_sem_proj) + F.mse_loss(z_qwen, z_sem_proj)

                    # Total
                    loss = (loss_gen_llama + loss_gen_qwen) + 0.5 * loss_align + 0.1 * loss_sem

                    batch_gen_loss += (loss_gen_llama.item() + loss_gen_qwen.item())
                    batch_align_loss += loss_align.item()
                    batch_sem_loss += loss_sem.item()
                else:
                    loss = loss_gen_llama
                    batch_gen_loss += loss_gen_llama.item()

                batch_loss += loss.item()
                loss = loss / args.batch_size  # Normalize for accumulation
                loss.backward()

            # Gradient step
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # Log
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                avg_loss = batch_loss / args.batch_size
                avg_gen = batch_gen_loss / args.batch_size
                print(f"  Step {step+1}/{args.steps}: loss={avg_loss:.4f} gen={avg_gen:.4f}", end="")
                if args.test_qwen and qwen_model is not None:
                    avg_align = batch_align_loss / args.batch_size
                    avg_sem = batch_sem_loss / args.batch_size
                    print(f" align={avg_align:.4f} sem={avg_sem:.4f}")
                else:
                    print()

        print(f"  ✓ Training complete")

        # ========================================================================
        # 7. Test generation AFTER training
        # ========================================================================

        print(f"\n[7/7] Testing generation after training...")

        alignment_tf.eval()
        adapter_llama.eval()
        if adapter_qwen is not None:
            adapter_qwen.eval()

        # Test on first 5 examples
        test_examples = examples[:min(5, len(examples))]

        print("\n  Results:")
        print("  " + "=" * 76)

        for i, ex in enumerate(test_examples):
            text = ex['source']
            answer = ex['answer']

            # Encode
            with torch.no_grad():
                z_sem = torch.tensor(
                    sem_encoder.encode([text], convert_to_tensor=False, show_progress_bar=False),
                    dtype=torch.float32,
                    device=device,
                )

                llama_tokens = llama_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                llama_input_ids = llama_tokens.input_ids.to(device)
                llama_attn_mask = llama_tokens.attention_mask.to(device)
                llama_embeds = llama_model.get_input_embeddings()(llama_input_ids)

                z_llama = alignment_tf(llama_embeds, z_sem, 'llama', llama_attn_mask)
                prefix_llama = adapter_llama(z_llama)

            # Generate
            pred = generate_from_prefix(
                llama_model,
                llama_tokenizer,
                prefix_llama,
                anchor_text="Answer: ",
                max_new_tokens=12,
            )

            print(f"  [{i+1}] Gold: {answer}")
            print(f"      Pred: {pred}")
            print()

        print("  " + "=" * 76)

        # Check if predictions are diverse (not collapsed)
        preds = []
        for ex in test_examples:
            text = ex['source']
            with torch.no_grad():
                z_sem = torch.tensor(
                    sem_encoder.encode([text], convert_to_tensor=False, show_progress_bar=False),
                    dtype=torch.float32,
                    device=device,
                )
                llama_tokens = llama_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                llama_input_ids = llama_tokens.input_ids.to(device)
                llama_attn_mask = llama_tokens.attention_mask.to(device)
                llama_embeds = llama_model.get_input_embeddings()(llama_input_ids)
                z_llama = alignment_tf(llama_embeds, z_sem, 'llama', llama_attn_mask)
                prefix_llama = adapter_llama(z_llama)
            pred = generate_from_prefix(llama_model, llama_tokenizer, prefix_llama, anchor_text="Answer: ", max_new_tokens=12)
            preds.append(pred)

        unique_preds = len(set(preds))
        print(f"\n  Diversity check: {unique_preds}/{len(preds)} unique predictions")
        if unique_preds == 1:
            print(f"  ⚠️  WARNING: All predictions are identical (collapsed): '{preds[0]}'")
        elif unique_preds < len(preds) * 0.5:
            print(f"  ⚠️  WARNING: Low diversity ({unique_preds}/{len(preds)})")
        else:
            print(f"  ✓ Good diversity!")
    else:
        print("\n[6/7] Skipping training (--steps 0)")
        print("[7/7] Skipping post-training evaluation")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nKey findings:")
    print(f"  ✓ All components instantiate correctly")
    print(f"  ✓ Forward pass works (z_llama: {z_llama.shape})")
    print(f"  ✓ Loss computation succeeds")
    if args.steps > 0:
        print(f"  ✓ Training loop runs ({args.steps} steps)")
        print(f"  ✓ Post-training predictions: {unique_preds}/{len(test_examples)} unique")
    print(f"\nTotal trainable parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Architecture: d_inter={args.d_inter}, num_slots={args.num_slots}")
    print("\nNext steps:")
    print("  1. Run longer training: --samples 1000 --steps 500")
    print("  2. Enable Qwen testing: --test_qwen")
    print("  3. Full experiment: --samples 10000 --steps 5000 --test_qwen")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test new interlingua architecture")
    parser.add_argument("--samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps (0 to skip)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_inter", type=int, default=512, help="Interlingua dimension")
    parser.add_argument("--num_slots", type=int, default=32, help="Number of soft token slots")
    parser.add_argument("--test_qwen", action="store_true", help="Also test with Qwen (slower)")

    args = parser.parse_args()

    test_architecture(args)
