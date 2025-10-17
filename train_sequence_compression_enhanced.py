#!/usr/bin/env python3
"""
Enhanced Sequence Compression Training Script

Tests 4 supervision mechanisms individually:
1. Contrastive diversity loss (InfoNCE)
2. K-token cross-entropy (focused supervision)
3. Reconstruction loss (information preservation)
4. Knowledge distillation (text teacher)

Each feature can be enabled via command-line flags for systematic comparison.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from latentwire.data import load_examples
from latentwire.core_utils import batch_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# Core Architecture (from original train_sequence_compression.py)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for preserving position information."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: [batch, seq] integer positions"""
        return self.pe[positions]


class LearnedAttentionPooling(nn.Module):
    """Learned cross-attention pooling with positional awareness."""

    def __init__(self, input_dim: int, target_length: int, num_heads: int = 8):
        super().__init__()
        self.target_length = target_length
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Learned queries for each output position
        self.queries = nn.Parameter(torch.randn(target_length, input_dim))

        # Positional encodings
        self.pos_encoding = PositionalEncoding(input_dim)

        # Attention projections
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, src_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, src_seq, input_dim] - input embeddings
            src_positions: [batch, src_seq] - original token positions

        Returns:
            [batch, target_length, input_dim] - compressed sequence
        """
        batch_size, src_seq, _ = x.shape

        # Add positional information to inputs
        pos_emb = self.pos_encoding(src_positions)  # [batch, src_seq, input_dim]
        x_with_pos = x + pos_emb

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, target_len, dim]

        # Add positional encoding to queries (linearly spaced positions)
        target_positions = torch.linspace(0, src_seq-1, self.target_length, device=x.device)
        target_positions = target_positions.long().unsqueeze(0).expand(batch_size, -1)
        target_pos_emb = self.pos_encoding(target_positions)
        queries = queries + target_pos_emb

        # Multi-head attention
        Q = self.q_proj(queries)  # [batch, target_len, dim]
        K = self.k_proj(x_with_pos)  # [batch, src_seq, dim]
        V = self.v_proj(x_with_pos)  # [batch, src_seq, dim]

        # Reshape for multi-head
        Q = Q.view(batch_size, self.target_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, target_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.target_length, self.input_dim)

        # Output projection and norm
        output = self.out_proj(attn_output)
        output = self.norm(output)

        return output


class SequenceCompressor(nn.Module):
    """Main module for sequence compression."""

    def __init__(
        self,
        input_dim: int,
        target_length: int,
        pooling_method: str = "learned_attention",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_length = target_length
        self.pooling_method = pooling_method

        if pooling_method == "learned_attention":
            self.pooler = LearnedAttentionPooling(input_dim, target_length)
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

    def forward(self, embeddings: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq, dim]
            positions: [batch, seq] - token positions
        Returns:
            [batch, target_length, dim]
        """
        return self.pooler(embeddings, positions)


# ============================================================================
# NEW FEATURE 1: Reconstruction Decoder
# ============================================================================

class ReconstructionDecoder(nn.Module):
    """Decoder for reconstructing source embeddings from compressed representation."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, compressed: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            compressed: [batch, compressed_len, dim] - compressed representations
            target_len: int - length to reconstruct to

        Returns:
            [batch, target_len, dim] - reconstructed embeddings
        """
        batch_size, compressed_len, dim = compressed.shape
        device = compressed.device

        # Create learnable query embeddings for target positions
        queries = torch.zeros(batch_size, target_len, dim, device=device)

        # Add positional encoding
        pos = torch.arange(target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_encoding = torch.sin(pos.unsqueeze(-1) / (10000 ** (torch.arange(0, dim, 2, device=device) / dim)))
        queries[:, :, 0::2] = pos_encoding
        if dim % 2 == 1:
            queries[:, :, 1::2] = pos_encoding[:, :, :-1]
        else:
            queries[:, :, 1::2] = pos_encoding

        # Decode
        reconstructed = self.decoder(queries, compressed)
        reconstructed = self.norm(reconstructed)

        return reconstructed


# ============================================================================
# NEW FEATURE 2: Contrastive Diversity Loss
# ============================================================================

def contrastive_diversity_loss(compressed_embeds: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss to prevent mode collapse.
    Forces different inputs to produce different compressed representations.

    Args:
        compressed_embeds: [batch, seq, dim] compressed representations
        temperature: Temperature for InfoNCE (default: 0.07 from CLIP)

    Returns:
        Scalar loss
    """
    batch_size, seq_len, dim = compressed_embeds.shape

    # Mean pool to get per-example representations
    pooled = compressed_embeds.mean(dim=1)  # [batch, dim]

    # Normalize
    pooled = F.normalize(pooled, dim=-1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(pooled, pooled.t()) / temperature  # [batch, batch]

    # Diagonal should be high (same example), off-diagonal low (different examples)
    labels = torch.arange(batch_size, device=compressed_embeds.device)

    # InfoNCE loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


# ============================================================================
# NEW FEATURE 3: Simplified K-token Cross-Entropy
# ============================================================================

def k_token_cross_entropy(
    model: nn.Module,
    compressed: torch.Tensor,
    answer_ids: torch.Tensor,
    K: int = 4,
    pad_token_id: int = -100,
) -> torch.Tensor:
    """
    Simplified K-token cross-entropy for diagnostic script.
    Supervises the first K tokens after the compressed prefix.

    Args:
        model: LLM model
        compressed: [batch, compressed_len, dim] compressed prefix
        answer_ids: [batch, answer_len] gold answer token IDs
        K: Number of tokens to supervise
        pad_token_id: Token ID to ignore

    Returns:
        Average loss over first K tokens
    """
    device = compressed.device
    batch_size = compressed.shape[0]
    answer_len = answer_ids.shape[1]

    total_loss = 0.0
    steps = 0

    for t in range(min(K, answer_len - 1)):  # -1 because we need t+1 for labels
        # Prepare inputs: compressed prefix + answer tokens up to t
        if t == 0:
            # First token: just compressed prefix
            inputs_embeds = compressed
        else:
            # Get embeddings for answer tokens [0:t]
            prev_answer_embeds = model.get_input_embeddings()(answer_ids[:, :t])
            inputs_embeds = torch.cat([compressed, prev_answer_embeds], dim=1)

        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds)
        logits = outputs.logits[:, -1, :]  # [batch, vocab] - last position

        # Target: next token
        target = answer_ids[:, t]

        # Compute loss (masking padding)
        loss_step = F.cross_entropy(
            logits,
            target,
            ignore_index=pad_token_id,
            reduction='mean'
        )

        total_loss += loss_step
        steps += 1

    return total_loss / max(steps, 1)


# ============================================================================
# NEW FEATURE 4: Knowledge Distillation (simplified)
# ============================================================================

@torch.no_grad()
def get_text_teacher_logits(
    model: nn.Module,
    tokenizer,
    sources: List[str],
    answers: List[str],
    K: int = 4,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Get teacher logits from text baseline for first K tokens.

    Returns:
        [batch, K, vocab_size] teacher logits
    """
    # Tokenize full text prompts
    full_texts = [f"{src} {ans}" for src, ans in zip(sources, answers)]
    encoded = tokenizer(
        full_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    # Forward pass
    outputs = model(**encoded)
    logits = outputs.logits  # [batch, seq, vocab]

    # Extract logits for first K answer tokens
    # Find where answers start (after source)
    source_lengths = []
    for src in sources:
        src_encoded = tokenizer(src, return_tensors='pt')
        source_lengths.append(src_encoded['input_ids'].shape[1])

    # Collect logits for first K answer positions
    batch_size = len(sources)
    vocab_size = logits.shape[-1]
    teacher_logits = torch.zeros(batch_size, K, vocab_size, device=device)

    for i, src_len in enumerate(source_lengths):
        for k in range(K):
            pos = src_len + k
            if pos < logits.shape[1]:
                teacher_logits[i, k] = logits[i, pos - 1]  # -1 for next-token prediction

    return teacher_logits


def knowledge_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    KL divergence loss for knowledge distillation.

    Args:
        student_logits: [batch, vocab] or [batch, K, vocab]
        teacher_logits: [batch, vocab] or [batch, K, vocab]
        tau: Temperature for softening distributions

    Returns:
        KL(teacher || student) loss
    """
    # Apply temperature
    student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1)

    # KL divergence
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    )

    # Scale by tau^2 (standard KD practice)
    return kl_div * (tau * tau)


# ============================================================================
# Dataset and Training
# ============================================================================

def log_diagnostics(log_file: Optional[str], step: int, epoch: int, metrics: Dict[str, Any]):
    """Log metrics to JSONL file."""
    if log_file is None or log_file == "":
        return

    entry = {
        "step": step,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        **metrics
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


class CompressionDataset(Dataset):
    """Dataset for sequence compression training."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Tokenize source
        source_encoded = self.tokenizer(
            ex['source'],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        # Tokenize answer
        answer_encoded = self.tokenizer(
            ex['answer'],
            return_tensors='pt',
            truncation=True,
            max_length=64,
            padding=False
        )

        return {
            'source': ex['source'],
            'answer': ex['answer'],
            'source_ids': source_encoded['input_ids'][0],
            'answer_ids': answer_encoded['input_ids'][0],
        }


def collate_fn(batch, pad_token_id: int):
    """Collate with padding."""
    max_src_len = max(item['source_ids'].shape[0] for item in batch)
    max_ans_len = max(item['answer_ids'].shape[0] for item in batch)

    batch_size = len(batch)
    source_ids = torch.full((batch_size, max_src_len), pad_token_id, dtype=torch.long)
    answer_ids = torch.full((batch_size, max_ans_len), pad_token_id, dtype=torch.long)
    src_mask = torch.zeros(batch_size, max_src_len)
    ans_mask = torch.zeros(batch_size, max_ans_len)

    sources = []
    answers = []

    for i, item in enumerate(batch):
        src_len = item['source_ids'].shape[0]
        ans_len = item['answer_ids'].shape[0]

        source_ids[i, :src_len] = item['source_ids']
        answer_ids[i, :ans_len] = item['answer_ids']
        src_mask[i, :src_len] = 1
        ans_mask[i, :ans_len] = 1

        sources.append(item['source'])
        answers.append(item['answer'])

    return {
        'source_ids': source_ids,
        'answer_ids': answer_ids,
        'src_mask': src_mask,
        'ans_mask': ans_mask,
        'sources': sources,
        'answers': answers,
    }


def compute_diversity(predictions: List[str]) -> float:
    """
    Compute diversity: percentage of unique predictions.

    Returns:
        diversity score in [0, 1]
    """
    if len(predictions) == 0:
        return 0.0

    unique_preds = set(predictions)
    return len(unique_preds) / len(predictions)


def train_epoch(
    compressor: nn.Module,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    diagnostic_log: str,
    tokenizer,
    args,  # Contains feature flags
    reconstruction_decoder: Optional[nn.Module] = None,
) -> int:
    """Train for one epoch with configurable losses."""
    compressor.train()
    if hasattr(model, 'enable_adapter_layers'):
        model.train()  # LoRA layers
    else:
        model.eval()  # Frozen base

    if reconstruction_decoder is not None:
        reconstruction_decoder.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_contrastive_loss = 0.0
    total_k_token_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kd_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        source_ids = batch['source_ids'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        src_mask = batch['src_mask'].to(device)
        ans_mask = batch['ans_mask'].to(device)
        sources = batch['sources']
        answers = batch['answers']

        # Get source embeddings
        with torch.no_grad():
            source_embeds = model.get_input_embeddings()(source_ids)  # [batch, src_seq, dim]

        # Create position tensor
        batch_size, src_seq = source_ids.shape
        positions = torch.arange(src_seq, device=device).unsqueeze(0).expand(batch_size, -1)

        # Compress sequence
        compressed = compressor(source_embeds, positions)  # [batch, target_len, dim]

        # ====================================================================
        # LOSS COMPUTATION
        # ====================================================================

        loss = 0.0
        loss_components = {}

        # BASELINE: Standard cross-entropy (always compute for logging)
        answer_embeds = model.get_input_embeddings()(answer_ids[:, :-1])  # [batch, ans_seq-1, dim]
        inputs_embeds = torch.cat([compressed, answer_embeds], dim=1)  # [batch, target_len + ans_seq-1, dim]

        # Create labels: mask compressed prefix, predict answer
        target_len = compressed.shape[1]
        labels = torch.full(
            (batch_size, target_len + answer_ids.shape[1] - 1),
            -100,
            dtype=torch.long,
            device=device
        )
        labels[:, target_len:] = answer_ids[:, 1:]  # Shift for next-token prediction

        # Mask padding in labels
        ans_mask_shifted = ans_mask[:, 1:]
        labels[:, target_len:][ans_mask_shifted == 0] = -100

        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        ce_loss = outputs.loss

        # Add CE loss if not using K-token (which replaces it)
        if not args.use_k_token_ce:
            loss = loss + ce_loss
            loss_components['ce'] = ce_loss.item()
        else:
            loss_components['ce_baseline'] = ce_loss.item()  # Log but don't use

        # FEATURE 1: Contrastive Diversity Loss
        if args.use_contrastive:
            contrastive_loss = contrastive_diversity_loss(compressed, args.contrastive_temp)
            loss = loss + args.contrastive_weight * contrastive_loss
            loss_components['contrastive'] = contrastive_loss.item()
            total_contrastive_loss += contrastive_loss.item()

        # FEATURE 2: K-token Cross-Entropy
        if args.use_k_token_ce:
            k_token_loss = k_token_cross_entropy(
                model,
                compressed,
                answer_ids,
                K=args.k_token_k,
                pad_token_id=tokenizer.pad_token_id,
            )
            loss = loss + k_token_loss
            loss_components['k_token'] = k_token_loss.item()
            total_k_token_loss += k_token_loss.item()

        # FEATURE 3: Reconstruction Loss
        if args.use_reconstruction and reconstruction_decoder is not None:
            reconstructed = reconstruction_decoder(compressed, src_seq)
            reconstruction_loss = F.mse_loss(reconstructed, source_embeds)
            loss = loss + args.reconstruction_weight * reconstruction_loss
            loss_components['reconstruction'] = reconstruction_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()

        # FEATURE 4: Knowledge Distillation
        if args.use_kd:
            # Get teacher logits (batched to save memory)
            with torch.no_grad():
                teacher_logits = get_text_teacher_logits(
                    model,
                    tokenizer,
                    sources,
                    answers,
                    K=args.kd_k,
                    device=device,
                )

            # Get student logits for first K tokens
            student_logits_list = []
            for t in range(min(args.kd_k, answer_ids.shape[1] - 1)):
                if t == 0:
                    inputs_embeds_t = compressed
                else:
                    prev_answer_embeds = model.get_input_embeddings()(answer_ids[:, :t])
                    inputs_embeds_t = torch.cat([compressed, prev_answer_embeds], dim=1)

                outputs_t = model(inputs_embeds=inputs_embeds_t)
                student_logits_list.append(outputs_t.logits[:, -1, :])

            if len(student_logits_list) > 0:
                student_logits = torch.stack(student_logits_list, dim=1)  # [batch, K, vocab]
                kd_loss = knowledge_distillation_loss(
                    student_logits,
                    teacher_logits[:, :len(student_logits_list), :],
                    tau=args.kd_tau,
                )
                loss = loss + args.kd_weight * kd_loss
                loss_components['kd'] = kd_loss.item()
                total_kd_loss += kd_loss.item()

        # ====================================================================
        # BACKWARD PASS
        # ====================================================================

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
        if hasattr(model, 'enable_adapter_layers'):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
        if reconstruction_decoder is not None:
            torch.nn.utils.clip_grad_norm_(reconstruction_decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            **{k: f'{v:.4f}' for k, v in loss_components.items() if k != 'ce_baseline'}
        })

        # Log every 50 steps
        if global_step % 50 == 0:
            log_diagnostics(diagnostic_log, global_step, epoch, {
                'type': 'train_step',
                'total_loss': loss.item(),
                **loss_components,
            })

        global_step += 1

    # Log epoch summary
    num_batches = len(dataloader)
    epoch_metrics = {
        'type': 'train_epoch',
        'avg_loss': total_loss / num_batches,
        'avg_ce_loss': total_ce_loss / num_batches,
    }

    if args.use_contrastive:
        epoch_metrics['avg_contrastive_loss'] = total_contrastive_loss / num_batches
    if args.use_k_token_ce:
        epoch_metrics['avg_k_token_loss'] = total_k_token_loss / num_batches
    if args.use_reconstruction:
        epoch_metrics['avg_reconstruction_loss'] = total_reconstruction_loss / num_batches
    if args.use_kd:
        epoch_metrics['avg_kd_loss'] = total_kd_loss / num_batches

    log_diagnostics(diagnostic_log, global_step, epoch, epoch_metrics)

    return global_step


@torch.no_grad()
def evaluate(
    compressor: nn.Module,
    model: nn.Module,
    tokenizer,
    dataset: List[Dict],
    device: torch.device,
    max_new_tokens: int = 12,
    batch_size: int = 8,
    max_length: int = 512,
    show_samples: int = 10,
) -> Dict[str, float]:
    """Evaluate compressed sequence generation with diversity metrics."""
    compressor.eval()
    model.eval()

    predictions = []
    references = []

    print(f"\nEvaluating on {len(dataset)} examples...")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Eval"):
        batch = dataset[i:i+batch_size]
        sources = [ex['source'] for ex in batch]

        # Tokenize
        encoded = tokenizer(
            sources,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )

        source_ids = encoded['input_ids'].to(device)

        # Get embeddings
        source_embeds = model.get_input_embeddings()(source_ids)

        # Create positions
        batch_size_actual, src_seq = source_ids.shape
        positions = torch.arange(src_seq, device=device).unsqueeze(0).expand(batch_size_actual, -1)

        # Compress
        compressed = compressor(source_embeds, positions)

        # Generate
        outputs = model.generate(
            inputs_embeds=compressed,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode
        for j, output in enumerate(outputs):
            pred_text = tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(batch[j]['answer'])

    # Compute metrics
    em, f1 = batch_metrics(predictions, references)
    diversity = compute_diversity(predictions)

    # Print sample predictions
    print(f"\n{'='*80}")
    print(f"SAMPLE PREDICTIONS (first {show_samples})")
    print('='*80)
    for i in range(min(show_samples, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"  Gold:       {references[i]}")
        print(f"  Prediction: {predictions[i]}")
    print('='*80)
    print(f"\nDiversity: {diversity:.1%} ({len(set(predictions))}/{len(predictions)} unique)")
    print(f"EM: {em:.2%}, F1: {f1:.2%}")
    print('='*80)

    return {
        'em': em,
        'f1': f1,
        'diversity': diversity,
        'predictions': predictions,
        'references': references,
    }


def main():
    parser = argparse.ArgumentParser(description="Enhanced sequence compression with feature flags")

    # Model
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')

    # Compression
    parser.add_argument('--target_sequence_length', type=int, default=128,
                       help='Target compressed sequence length')
    parser.add_argument('--source_length', type=int, default=300,
                       help='Expected average source sequence length (for reporting compression ratio only)')
    parser.add_argument('--pooling_method', type=str, default='learned_attention',
                       choices=['learned_attention'])

    # LoRA
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA adaptation')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_layers', type=int, default=None,
                       help='Apply LoRA to first N layers (None = all layers)')

    # Training
    parser.add_argument('--dataset', type=str, default='squad', choices=['squad', 'hotpot'])
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-4)

    # Evaluation
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=12)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--show_samples', type=int, default=10,
                       help='Number of sample predictions to show during eval')

    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================

    # Feature 1: Contrastive Diversity Loss
    parser.add_argument('--use_contrastive', action='store_true',
                       help='Enable contrastive diversity loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                       help='Weight for contrastive loss')
    parser.add_argument('--contrastive_temp', type=float, default=0.07,
                       help='Temperature for contrastive loss (CLIP default: 0.07)')

    # Feature 2: K-token Cross-Entropy
    parser.add_argument('--use_k_token_ce', action='store_true',
                       help='Enable K-token cross-entropy (replaces standard CE)')
    parser.add_argument('--k_token_k', type=int, default=4,
                       help='Number of tokens to supervise with K-token CE')

    # Feature 3: Reconstruction Loss
    parser.add_argument('--use_reconstruction', action='store_true',
                       help='Enable reconstruction loss')
    parser.add_argument('--reconstruction_weight', type=float, default=0.1,
                       help='Weight for reconstruction loss')
    parser.add_argument('--reconstruction_layers', type=int, default=2,
                       help='Number of transformer decoder layers for reconstruction')

    # Feature 4: Knowledge Distillation
    parser.add_argument('--use_kd', action='store_true',
                       help='Enable knowledge distillation from text teacher')
    parser.add_argument('--kd_weight', type=float, default=0.3,
                       help='Weight for KD loss')
    parser.add_argument('--kd_tau', type=float, default=1.0,
                       help='Temperature for KD')
    parser.add_argument('--kd_k', type=int, default=4,
                       help='Number of tokens for KD')

    # I/O
    parser.add_argument('--save_dir', type=str, default='./runs/seq_compression')
    parser.add_argument('--diagnostic_log', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.diagnostic_log:
        Path(args.diagnostic_log).parent.mkdir(parents=True, exist_ok=True)

    # Setup device
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/dev/null'
    os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/dev/null'

    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"Using device: {device}")
            print(f"GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            raise RuntimeError("CUDA not available")
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        device = torch.device('cpu')

    print()

    # Print feature configuration
    print("="*80)
    print("FEATURE CONFIGURATION")
    print("="*80)
    print(f"Baseline CE loss: {'No (using K-token)' if args.use_k_token_ce else 'Yes'}")
    print(f"Contrastive diversity: {args.use_contrastive} (weight={args.contrastive_weight}, temp={args.contrastive_temp})")
    print(f"K-token CE: {args.use_k_token_ce} (K={args.k_token_k})")
    print(f"Reconstruction: {args.use_reconstruction} (weight={args.reconstruction_weight}, layers={args.reconstruction_layers})")
    print(f"Knowledge distillation: {args.use_kd} (weight={args.kd_weight}, tau={args.kd_tau}, K={args.kd_k})")
    print("="*80)
    print()

    # Load model
    print(f"Loading {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(device)

    embed_device = model.get_input_embeddings().weight.device
    model_dtype = model.get_input_embeddings().weight.dtype
    d_model = model.config.hidden_size

    # Apply LoRA if requested
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        print(f"\nApplying LoRA: r={args.lora_r}, alpha={args.lora_alpha}, layers={args.lora_layers or 'all'}")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=list(range(args.lora_layers)) if args.lora_layers else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print()
        model.train()  # Enable LoRA training
    else:
        model.eval()  # Frozen

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print(f"Model loaded!\n")

    # Create compressor
    print(f"Creating sequence compressor: {args.source_length} → {args.target_sequence_length}")
    print(f"Pooling method: {args.pooling_method}")
    print(f"Compression ratio: {args.source_length / args.target_sequence_length:.2f}×\n")

    compressor = SequenceCompressor(
        input_dim=d_model,
        target_length=args.target_sequence_length,
        pooling_method=args.pooling_method,
    ).to(embed_device).to(model_dtype)

    num_params = sum(p.numel() for p in compressor.parameters())
    print(f"Compressor parameters: {num_params:,}\n")

    # Create reconstruction decoder if needed
    reconstruction_decoder = None
    if args.use_reconstruction:
        print(f"Creating reconstruction decoder ({args.reconstruction_layers} layers)...")
        reconstruction_decoder = ReconstructionDecoder(
            hidden_dim=d_model,
            num_layers=args.reconstruction_layers,
            num_heads=8,
        ).to(embed_device).to(model_dtype)

        recon_params = sum(p.numel() for p in reconstruction_decoder.parameters())
        print(f"Reconstruction decoder parameters: {recon_params:,}\n")

    # Load data
    print(f"Loading {args.samples} training examples from {args.dataset}...")
    train_examples = load_examples(
        dataset=args.dataset,
        split='train',
        samples=args.samples,
        seed=args.seed
    )

    print(f"Loading {args.eval_samples} eval examples...")
    eval_examples = load_examples(
        dataset=args.dataset,
        split='validation',
        samples=args.eval_samples,
        seed=args.seed + 1
    )
    print(f"Loaded {len(train_examples)} train, {len(eval_examples)} eval\n")

    # Create dataset
    train_dataset = CompressionDataset(train_examples, tokenizer, args.max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    # Optimizer
    trainable_params = list(compressor.parameters())
    if args.use_lora:
        trainable_params += [p for p in model.parameters() if p.requires_grad]
    if reconstruction_decoder is not None:
        trainable_params += list(reconstruction_decoder.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Training loop
    print("="*80)
    print("TRAINING")
    print("="*80)
    print()

    best_f1 = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}\n")

        # Train
        global_step = train_epoch(
            compressor=compressor,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=embed_device,
            epoch=epoch,
            global_step=global_step,
            diagnostic_log=args.diagnostic_log,
            tokenizer=tokenizer,
            args=args,
            reconstruction_decoder=reconstruction_decoder,
        )

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nEvaluating epoch {epoch + 1}...")
            eval_metrics = evaluate(
                compressor=compressor,
                model=model,
                tokenizer=tokenizer,
                dataset=eval_examples,
                device=embed_device,
                max_new_tokens=args.max_new_tokens,
                batch_size=8,
                max_length=args.max_length,
                show_samples=args.show_samples,
            )

            print(f"\nEval Results:")
            print(f"  EM: {eval_metrics['em']:.2%}")
            print(f"  F1: {eval_metrics['f1']:.2%}")
            print(f"  Diversity: {eval_metrics['diversity']:.1%}")

            # Log
            log_diagnostics(args.diagnostic_log, global_step, epoch, {
                'type': 'full_eval',
                'em': eval_metrics['em'],
                'f1': eval_metrics['f1'],
                'diversity': eval_metrics['diversity'],
            })

            # Save best
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']

                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'compressor_state_dict': compressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'config': vars(args),
                }

                if args.use_lora:
                    checkpoint['lora_state_dict'] = model.state_dict()
                if reconstruction_decoder is not None:
                    checkpoint['reconstruction_decoder_state_dict'] = reconstruction_decoder.state_dict()

                ckpt_path = save_dir / 'best_checkpoint.pt'
                torch.save(checkpoint, ckpt_path)
                print(f"  → Saved best checkpoint: {ckpt_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest F1: {best_f1:.2%}")
    print(f"Compression ratio: {args.source_length / args.target_sequence_length:.2f}×")
    print(f"Results saved to: {save_dir}")
    print()


if __name__ == '__main__':
    main()
