#!/usr/bin/env python3
"""
SQuAD Compression Ablations: Test different compression ratios, architectures, and losses.
Designed for 4 GPU parallel execution with comprehensive ablations.

Key experiments:
- Compression ratios: 32, 64, 128 tokens
- Architectures: Cross-attention, Convolution, Weighted pooling
- Loss functions: Different weightings of KD, CE, contrastive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from datasets import load_dataset
try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT library not available. LoRA training will be disabled.")
    print("Install with: pip install peft")
import json
import time
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import multiprocessing as mp
from tqdm import tqdm
import math

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CompressionConfig:
    """Configuration for compression experiments."""
    # Model
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Compression
    target_length: int = 64  # M: number of compressed tokens
    architecture: str = "cross_attention"  # cross_attention, conv, pooling, gist

    # Training
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    lora_lr: float = 5e-5
    epochs: int = 10
    warmup_ratio: float = 0.05

    # Loss weights
    loss_weights: Dict[str, float] = None

    # Data
    num_train_samples: int = 10000
    num_eval_samples: int = 1000
    max_input_length: int = 512

    # System
    device: str = "cuda"
    gpu_id: int = 0
    use_bf16: bool = True
    seed: int = 42
    output_dir: str = "runs/compression_ablations"

    def __post_init__(self):
        if self.loss_weights is None:
            # Default: Balanced between teacher-forcing and generation
            self.loss_weights = {
                'teacher_forcing': 0.5,
                'generation': 0.3,
                'contrastive': 0.2
            }

# ============================================================================
# Compression Architectures
# ============================================================================

class CrossAttentionCompressor(nn.Module):
    """
    Compress via learned cross-attention queries.
    Most flexible - can select information from anywhere.
    """
    def __init__(self, target_length=64, hidden_dim=4096):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim

        # Learnable queries that will extract information
        self.queries = nn.Parameter(torch.randn(target_length, hidden_dim) * 0.02)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=32,  # Llama-3.1 8B has 32 heads
            batch_first=True,
            dropout=0.1
        )

        # Layer norm and residual refinement
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Small MLP for refinement
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Initialize MLP near identity
        with torch.no_grad():
            self.mlp[-1].weight.data *= 0.01
            self.mlp[-1].bias.data.zero_()

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, 4096] - full sequence embeddings
        Returns:
            compressed: [batch, target_length, 4096]
        """
        batch_size = full_embeds.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to full sequence
        compressed, attn_weights = self.cross_attn(
            query=queries,
            key=full_embeds,
            value=full_embeds
        )

        # Residual + norm
        compressed = self.norm1(compressed + queries)

        # MLP refinement
        refined = self.mlp(compressed)
        compressed = self.norm2(compressed + refined)

        return compressed, attn_weights


class ConvolutionalCompressor(nn.Module):
    """
    Compress via strided 1D convolution.
    Better for local patterns and maintaining order.
    """
    def __init__(self, target_length=64, hidden_dim=4096):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim

        # Calculate stride needed for compression
        # Assuming input ~512 tokens, we need stride ~512/target_length
        self.stride = max(1, 512 // target_length)
        self.kernel_size = self.stride * 2

        # Depthwise convolution (each channel processed separately)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=hidden_dim  # Depthwise
        )

        # Pointwise convolution to mix information
        self.pointwise = nn.Linear(hidden_dim, hidden_dim)

        # Adaptive pooling to exact target length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, 4096]
        Returns:
            compressed: [batch, target_length, 4096]
        """
        # Transpose for Conv1d: [batch, seq, hidden] -> [batch, hidden, seq]
        x = full_embeds.transpose(1, 2)

        # Convolutional compression
        compressed = self.conv(x)

        # Ensure exact target length
        compressed = self.adaptive_pool(compressed)

        # Transpose back: [batch, hidden, target_len] -> [batch, target_len, hidden]
        compressed = compressed.transpose(1, 2)

        # Mix information across dimensions
        compressed = self.pointwise(compressed)
        compressed = self.norm(compressed)

        return compressed, None  # No attention weights


class WeightedPoolingCompressor(nn.Module):
    """
    Compress via learned weighted pooling over windows.
    Simplest and most interpretable.
    """
    def __init__(self, target_length=64, hidden_dim=4096, max_seq_len=512):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Learn importance weights for each position in each window
        window_size = max_seq_len // target_length + 1
        self.importance_weights = nn.Parameter(
            torch.ones(target_length, window_size) / window_size
        )

        # Optional: learn different projections for each compressed position
        self.position_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(target_length)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, 4096]
        Returns:
            compressed: [batch, target_length, 4096]
        """
        batch_size, seq_len, hidden_dim = full_embeds.shape
        window_size = seq_len // self.target_length + 1

        compressed = []

        for i in range(self.target_length):
            # Define window boundaries
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, seq_len)
            window_len = end_idx - start_idx

            if window_len > 0:
                # Extract window
                window = full_embeds[:, start_idx:end_idx, :]

                # Apply importance weights
                weights = F.softmax(
                    self.importance_weights[i, :window_len], dim=0
                ).unsqueeze(0).unsqueeze(-1)

                # Weighted average
                pooled = (window * weights).sum(dim=1)

                # Position-specific projection
                pooled = self.position_projections[i](pooled)
                compressed.append(pooled)

        compressed = torch.stack(compressed, dim=1)
        compressed = self.norm(compressed)

        return compressed, self.importance_weights


class GistCompressor(nn.Module):
    """
    Compress via learnable "gist" tokens inserted into the sequence.

    Reference: "Learning to Compress Prompts with Gist Tokens" (Mu et al., NeurIPS 2023)
    Paper: https://arxiv.org/abs/2304.08467

    Key idea:
    - Insert learnable gist tokens at the beginning of the sequence
    - Gist tokens attend to full sequence and compress information
    - Rest of model only needs to attend to gist tokens
    - Trained via masked instruction finetuning

    Implementation:
    - Learnable gist token embeddings
    - Bidirectional attention (gist tokens can see everything)
    - Position-aware to maintain sequence order
    """
    def __init__(self, target_length=64, hidden_dim=4096):
        super().__init__()
        self.target_length = target_length  # Number of gist tokens
        self.hidden_dim = hidden_dim

        # Learnable gist token embeddings
        # Initialize similar to word embeddings
        self.gist_embeddings = nn.Parameter(
            torch.randn(target_length, hidden_dim) * 0.02
        )

        # Transformer layer for gist token refinement
        # Gist tokens attend to full sequence to compress information
        self.gist_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=32,  # Match Llama architecture
            batch_first=True,
            dropout=0.1
        )

        # Self-attention among gist tokens for refinement
        self.gist_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=32,
            batch_first=True,
            dropout=0.1
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Feed-forward network for gist token refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

        # Positional encoding for gist tokens
        self.gist_position_embed = nn.Parameter(
            torch.randn(target_length, hidden_dim) * 0.02
        )

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, hidden_dim] - full sequence embeddings
        Returns:
            compressed: [batch, target_length, hidden_dim] - gist tokens
            attn_weights: Attention weights showing what gist tokens attend to
        """
        batch_size = full_embeds.size(0)

        # Initialize gist tokens with learned embeddings + positional encoding
        gist_tokens = self.gist_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        gist_tokens = gist_tokens + self.gist_position_embed.unsqueeze(0)

        # Step 1: Gist tokens attend to full sequence (compression step)
        # Query: gist tokens, Key/Value: full sequence
        compressed, attn_weights = self.gist_attention(
            query=gist_tokens,
            key=full_embeds,
            value=full_embeds
        )
        compressed = self.norm1(compressed + gist_tokens)

        # Step 2: Self-attention among gist tokens (refinement step)
        # This allows gist tokens to exchange information
        refined, _ = self.gist_self_attention(
            query=compressed,
            key=compressed,
            value=compressed
        )
        compressed = self.norm2(compressed + refined)

        # Step 3: Feed-forward network for final refinement
        output = self.ffn(compressed)
        compressed = self.norm3(compressed + output)

        return compressed, attn_weights


# ============================================================================
# SQuAD Dataset Wrapper
# ============================================================================

class SQuADCompressionDataset(Dataset):
    """SQuAD dataset for compression training."""

    def __init__(self, split='train', num_samples=10000, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load SQuAD
        dataset = load_dataset('squad', split=split)

        # Sample if needed
        if num_samples and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)

        self.data = []
        for item in tqdm(dataset, desc=f"Processing {split} data"):
            # Format prompt
            prompt = f"Question: {item['question']}\nContext: {item['context']}\nAnswer:"
            answer = item['answers']['text'][0] if item['answers']['text'] else ""

            self.data.append({
                'prompt': prompt,
                'answer': answer,
                'question': item['question'],
                'context': item['context']
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt (NO padding - done in collate_fn)
        prompt_tokens = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Dynamic padding in batch
            return_tensors=None
        )

        # Tokenize answer
        answer_tokens = self.tokenizer(
            item['answer'],
            max_length=32,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        return {
            'input_ids': prompt_tokens['input_ids'],
            'attention_mask': prompt_tokens['attention_mask'],
            'answer_ids': answer_tokens['input_ids'],
            'answer_mask': answer_tokens['attention_mask'],
            'answer_text': item['answer']
        }

    @staticmethod
    def collate_fn(tokenizer):
        """Create collate function with dynamic padding."""
        def collate(batch):
            # Extract fields
            input_ids = [torch.tensor(item['input_ids']) for item in batch]
            attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
            answer_ids = [torch.tensor(item['answer_ids']) for item in batch]
            answer_masks = [torch.tensor(item['answer_mask']) for item in batch]
            answer_texts = [item['answer_text'] for item in batch]

            # Pad to max length in batch (not global max)
            from torch.nn.utils.rnn import pad_sequence
            input_ids_padded = pad_sequence(input_ids, batch_first=True,
                                           padding_value=tokenizer.pad_token_id)
            attention_masks_padded = pad_sequence(attention_masks, batch_first=True,
                                                 padding_value=0)
            answer_ids_padded = pad_sequence(answer_ids, batch_first=True,
                                            padding_value=tokenizer.pad_token_id)
            answer_masks_padded = pad_sequence(answer_masks, batch_first=True,
                                              padding_value=0)

            return {
                'input_ids': input_ids_padded,
                'attention_mask': attention_masks_padded,
                'answer_ids': answer_ids_padded,
                'answer_mask': answer_masks_padded,
                'answer_text': answer_texts
            }
        return collate


# ============================================================================
# Training Functions
# ============================================================================

class CompressionTrainer:
    """Trainer for compression models."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}")

        # Setup output directory
        self.output_dir = Path(config.output_dir) / f"M{config.target_length}_{config.architecture}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'teacher_forcing_loss': [],
            'generation_loss': [],
            'contrastive_loss': [],
            'eval_f1': [],
            'eval_em': [],
            'baseline_f1': None,
            'baseline_em': None,
            'compression_ratio': config.max_input_length / config.target_length,
            'sample_predictions': []
        }

    def setup_model(self):
        """Initialize model, tokenizer, and compressor."""
        print(f"Loading model: {self.config.model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            'torch_dtype': torch.bfloat16 if self.config.use_bf16 else torch.float32,
            'device_map': None,  # Manual placement for multi-GPU
            'low_cpu_mem_usage': True,
        }

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **model_kwargs
        ).to(self.device).eval()

        # Apply LoRA if available
        if PEFT_AVAILABLE:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            )

            self.model = get_peft_model(self.base_model, peft_config)
            self.model.print_trainable_parameters()
            print("LoRA applied successfully")
        else:
            print("WARNING: PEFT not available - training full model (not recommended)")
            self.model = self.base_model
            # Enable gradient for base model
            for param in self.model.parameters():
                param.requires_grad = True

        # Initialize compressor
        hidden_dim = self.base_model.config.hidden_size
        dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32

        if self.config.architecture == "cross_attention":
            self.compressor = CrossAttentionCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device, dtype=dtype)
        elif self.config.architecture == "conv":
            self.compressor = ConvolutionalCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device, dtype=dtype)
        elif self.config.architecture == "pooling":
            self.compressor = WeightedPoolingCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device, dtype=dtype)
        elif self.config.architecture == "gist":
            self.compressor = GistCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device, dtype=dtype)
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}. "
                           f"Choose from: cross_attention, conv, pooling, gist")

        # Get embedding layer
        self.embed_layer = self.base_model.get_input_embeddings()

    def setup_data(self):
        """Initialize datasets and dataloaders."""
        print("Loading SQuAD dataset...")

        # Training dataset
        self.train_dataset = SQuADCompressionDataset(
            split='train',
            num_samples=self.config.num_train_samples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_input_length
        )

        # Evaluation dataset
        self.eval_dataset = SQuADCompressionDataset(
            split='validation',
            num_samples=self.config.num_eval_samples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_input_length
        )

        # Create dataloaders with dynamic padding
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=SQuADCompressionDataset.collate_fn(self.tokenizer)
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size * 2,  # Can use larger batch for eval
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=SQuADCompressionDataset.collate_fn(self.tokenizer)
        )

    def compute_loss(self, batch):
        """Compute training loss with multiple components.

        CRITICAL FIXES:
        1. Teacher-forcing uses correct indexing (answer_start, not answer_start-1)
        2. KL distillation REMOVED (misaligned positions made it meaningless)
        3. Added generation loss to match eval objective
        4. Contrastive uses first token (CLS-like) instead of mean pooling
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        answer_ids = batch['answer_ids'].to(self.device)
        answer_mask = batch['answer_mask'].to(self.device)

        # Get full embeddings (frozen - no gradients)
        with torch.no_grad():
            full_embeds = self.embed_layer(input_ids)

        # Compress
        compressed, attn_weights = self.compressor(full_embeds)

        losses = {}

        # 1. Teacher-forcing loss on answer generation
        if self.config.loss_weights.get('teacher_forcing', 0) > 0:
            # Append answer tokens to compressed sequence
            with torch.no_grad():
                answer_embeds = self.embed_layer(answer_ids)

            combined = torch.cat([compressed, answer_embeds], dim=1)

            # Forward pass with combined sequence
            combined_outputs = self.model(inputs_embeds=combined)
            combined_logits = combined_outputs.logits

            # Compute loss on answer portion (FIXED: was answer_start-1, now answer_start)
            answer_start = compressed.size(1)
            answer_logits = combined_logits[:, answer_start:-1, :]  # Predict answer tokens

            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            teacher_forcing_loss = loss_fct(
                answer_logits.reshape(-1, answer_logits.size(-1)),
                answer_ids.reshape(-1)
            )
            losses['teacher_forcing'] = teacher_forcing_loss

        # 2. Generation loss (ADDED: matches eval objective)
        # Generate from compressed representation and compare to answers
        if self.config.loss_weights.get('generation', 0) > 0:
            # Generate greedily
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs_embeds=compressed,
                    max_new_tokens=answer_ids.size(1),
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Compute token-level accuracy loss
            # Get logits for generated sequence
            gen_embeds = self.embed_layer(generated_ids)
            gen_outputs = self.model(inputs_embeds=gen_embeds)
            gen_logits = gen_outputs.logits[:, -answer_ids.size(1):, :]

            generation_loss = loss_fct(
                gen_logits.reshape(-1, gen_logits.size(-1)),
                answer_ids.reshape(-1)
            )
            losses['generation'] = generation_loss

        # 3. Contrastive loss to prevent collapse (FIXED: use first token, not mean)
        if self.config.loss_weights.get('contrastive', 0) > 0 and compressed.size(0) > 1:
            # Use first token as representation (CLS-like)
            compressed_repr = compressed[:, 0, :]  # [batch, hidden_dim]
            compressed_norm = F.normalize(compressed_repr, p=2, dim=-1)

            # Compute pairwise similarities
            sim_matrix = torch.matmul(compressed_norm, compressed_norm.t())

            # Contrastive loss: push different samples apart
            batch_size = compressed.size(0)
            mask = torch.eye(batch_size).bool().to(self.device)

            # Mean similarity of different samples (should be low)
            off_diagonal = sim_matrix[~mask].mean()

            # Penalize high similarity between different samples
            contrastive_loss = torch.relu(off_diagonal - 0.3)  # Target max similarity of 0.3
            losses['contrastive'] = contrastive_loss

        # Combine losses (KL_DISTILL REMOVED - was broken)
        total_loss = sum(
            self.config.loss_weights.get(k, 0) * v
            for k, v in losses.items()
            if k in self.config.loss_weights
        )

        return total_loss, losses

    def train_epoch(self, epoch):
        """Train for one epoch with gradient clipping and detailed loss tracking."""
        self.model.train()
        self.compressor.train()

        epoch_losses = []
        epoch_component_losses = {
            'teacher_forcing': [],
            'generation': [],
            'contrastive': []
        }

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Compute loss
            loss, loss_dict = self.compute_loss(batch)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation
            loss.backward()

            # Update weights (ADDED: gradient clipping)
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.compressor.parameters()),
                    max_norm=1.0
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track metrics (unscaled loss)
            epoch_losses.append(loss.item() * self.config.gradient_accumulation)

            # Track per-component losses
            for key in epoch_component_losses:
                if key in loss_dict:
                    epoch_component_losses[key].append(loss_dict[key].item())

            # Update progress bar with component losses
            postfix = {
                'loss': f"{epoch_losses[-1]:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            }
            if epoch_component_losses['teacher_forcing']:
                postfix['tf'] = f"{epoch_component_losses['teacher_forcing'][-1]:.3f}"
            if epoch_component_losses['contrastive']:
                postfix['con'] = f"{epoch_component_losses['contrastive'][-1]:.3f}"

            progress_bar.set_postfix(postfix)

        # Save component losses to metrics
        for key, values in epoch_component_losses.items():
            if values:
                self.metrics[f'{key}_loss'].append(np.mean(values))

        return np.mean(epoch_losses)

    def evaluate(self, save_samples=True, num_samples=10):
        """Evaluate on validation set with sample logging."""
        self.model.eval()
        self.compressor.eval()

        all_predictions = []
        all_references = []
        sample_predictions = []

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)

                # Get compressed representation
                full_embeds = self.embed_layer(input_ids)
                compressed, _ = self.compressor(full_embeds)

                # Generate answer
                outputs = self.model.generate(
                    inputs_embeds=compressed,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Decode predictions
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                references = batch['answer_text']

                all_predictions.extend(predictions)
                all_references.extend(references)

                # Save first few samples
                if save_samples and len(sample_predictions) < num_samples:
                    for pred, ref in zip(predictions, references):
                        if len(sample_predictions) < num_samples:
                            sample_predictions.append({
                                'predicted': pred,
                                'reference': ref
                            })

        # Compute F1 and EM scores
        f1_scores = []
        em_scores = []

        for pred, ref in zip(all_predictions, all_references):
            # Simple F1 calculation
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # Use Counter for proper token-level F1
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)

            # Count overlapping tokens (minimum of counts for each token)
            common_count = sum((pred_counter & ref_counter).values())

            if common_count == 0:
                f1 = 0.0
            else:
                precision = common_count / len(pred_tokens) if pred_tokens else 0
                recall = common_count / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

            # Exact match
            em = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
            em_scores.append(em)

        if save_samples:
            self.metrics['sample_predictions'] = sample_predictions

        return np.mean(f1_scores), np.mean(em_scores)

    def evaluate_baseline(self):
        """Evaluate uncompressed teacher model to get baseline performance."""
        print("Evaluating baseline (uncompressed teacher model)...")
        self.base_model.eval()

        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Baseline eval"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Generate from FULL input (no compression)
                outputs = self.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Decode predictions
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                references = batch['answer_text']

                all_predictions.extend(predictions)
                all_references.extend(references)

        # Compute F1 and EM scores
        f1_scores = []
        em_scores = []

        for pred, ref in zip(all_predictions, all_references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # Use Counter for proper token-level F1
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)

            # Count overlapping tokens (minimum of counts for each token)
            common_count = sum((pred_counter & ref_counter).values())

            if common_count == 0:
                f1 = 0.0
            else:
                precision = common_count / len(pred_tokens) if pred_tokens else 0
                recall = common_count / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

            em = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
            em_scores.append(em)

        baseline_f1 = np.mean(f1_scores)
        baseline_em = np.mean(em_scores)

        # Save to metrics
        self.metrics['baseline_f1'] = baseline_f1
        self.metrics['baseline_em'] = baseline_em

        print(f"Baseline F1: {baseline_f1:.4f}, EM: {baseline_em:.4f}")

        return baseline_f1, baseline_em

    def train(self):
        """Main training loop with baseline evaluation and proper warmup."""
        # Setup reproducibility (ADDED)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Setup model and data
        self.setup_model()
        self.setup_data()

        # Evaluate baseline FIRST (ADDED)
        print("\n" + "="*80)
        print("BASELINE EVALUATION (Uncompressed Teacher)")
        print("="*80)
        self.evaluate_baseline()
        print("="*80 + "\n")

        # Initialize optimizer
        optimizer_groups = [
            {'params': self.compressor.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.parameters(), 'lr': self.config.lora_lr}
        ]
        self.optimizer = AdamW(optimizer_groups, weight_decay=0.01)

        # Learning rate scheduler with PROPER warmup (FIXED)
        num_training_steps = len(self.train_loader) * self.config.epochs // self.config.gradient_accumulation
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        print(f"Batches per epoch: {len(self.train_loader)}, Grad accum: {self.config.gradient_accumulation}\n")

        # Training loop
        best_f1 = 0.0

        for epoch in range(self.config.epochs):
            # Train
            avg_loss = self.train_epoch(epoch)
            self.metrics['train_loss'].append(avg_loss)

            # Evaluate
            f1_score, em_score = self.evaluate()
            self.metrics['eval_f1'].append(f1_score)
            self.metrics['eval_em'].append(em_score)

            # Compute % of baseline
            if self.metrics['baseline_f1'] is not None:
                pct_baseline = f1_score / self.metrics['baseline_f1'] * 100
            else:
                pct_baseline = 0

            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Eval F1: {f1_score:.4f} ({pct_baseline:.1f}% of baseline)")
            print(f"  Eval EM: {em_score:.4f}")

            # Save best model
            if f1_score > best_f1:
                best_f1 = f1_score
                self.save_checkpoint(epoch, is_best=True)

            # Save metrics after each epoch
            self.save_metrics()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'compressor_state': self.compressor.state_dict(),
            'model_state': self.model.state_dict(),
            'metrics': self.metrics
        }

        # Save checkpoint
        checkpoint_name = 'best_checkpoint.pt' if is_best else f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, self.output_dir / checkpoint_name)

        print(f"Saved {'best ' if is_best else ''}checkpoint to {self.output_dir / checkpoint_name}")

    def save_metrics(self):
        """Save training metrics."""
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# ============================================================================
# Parallel Ablation Runner
# ============================================================================

def run_single_ablation(config: CompressionConfig, gpu_id: int, result_queue=None):
    """Run a single ablation on specified GPU.

    FIXED: Results now properly returned via queue instead of lost in subprocess.
    """
    config.gpu_id = gpu_id

    # Set random seed (full reproducibility)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    print(f"\n{'='*80}")
    print(f"Starting ablation: M={config.target_length}, {config.architecture}")
    print(f"GPU: {gpu_id}, Loss weights: {config.loss_weights}")
    print(f"{'='*80}\n")

    try:
        trainer = CompressionTrainer(config)
        trainer.train()

        # Collect final metrics
        result = {
            'target_length': config.target_length,
            'architecture': config.architecture,
            'loss_weights': config.loss_weights,
            'final_f1': trainer.metrics['eval_f1'][-1],
            'final_em': trainer.metrics['eval_em'][-1],
            'best_f1': max(trainer.metrics['eval_f1']),
            'baseline_f1': trainer.metrics['baseline_f1'],
            'compression_ratio': trainer.metrics['compression_ratio']
        }

        # Put result in queue if provided
        if result_queue is not None:
            result_queue.put(result)

        return result

    except Exception as e:
        import traceback
        print(f"Error in ablation: {e}")
        print(traceback.format_exc())

        error_result = {
            'error': str(e),
            'target_length': config.target_length,
            'architecture': config.architecture
        }

        if result_queue is not None:
            result_queue.put(error_result)

        return error_result


def run_all_ablations():
    """Run all ablations across 4 GPUs.

    FIXED: Uses multiprocessing Queue to properly collect results.
    """
    from multiprocessing import Queue

    # Define ablation grid
    ablations = []

    # Compression ratios
    target_lengths = [32, 64, 128]

    # Architectures
    architectures = ["cross_attention", "conv", "pooling"]

    # Loss weight configurations (KL REMOVED - it was broken)
    loss_configs = [
        {'teacher_forcing': 0.8, 'generation': 0.0, 'contrastive': 0.2},  # Task-focused (TF only)
        {'teacher_forcing': 0.5, 'generation': 0.3, 'contrastive': 0.2},  # Balanced (TF + Gen)
        {'teacher_forcing': 0.0, 'generation': 0.8, 'contrastive': 0.2},  # Generation-focused
    ]

    # Create all combinations
    for target_length in target_lengths:
        for architecture in architectures:
            for loss_weights in loss_configs:
                config = CompressionConfig(
                    target_length=target_length,
                    architecture=architecture,
                    loss_weights=loss_weights
                )
                ablations.append(config)

    print(f"Total ablations to run: {len(ablations)}")
    print(f"Estimated time: {len(ablations) * 2} hours (assuming 2 hours per ablation)")

    # Run ablations in batches of 4 (one per GPU)
    all_results = []

    for i in range(0, len(ablations), 4):
        batch = ablations[i:i+4]
        processes = []
        result_queue = Queue()

        print(f"\n{'='*80}")
        print(f"Starting batch {i//4 + 1}/{(len(ablations) + 3)//4}")
        print(f"{'='*80}\n")

        # Start processes for this batch
        for j, config in enumerate(batch):
            p = mp.Process(
                target=run_single_ablation,
                args=(config, j % 4, result_queue)
            )
            p.start()
            processes.append(p)

        # Wait for batch to complete
        for p in processes:
            p.join()

        # Collect results from queue
        while not result_queue.empty():
            result = result_queue.get()
            all_results.append(result)

    # Save all results
    results_file = Path("runs/compression_ablations/all_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll ablations complete! Results saved to {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION SUMMARY (Top 10)")
    print("="*80)

    valid_results = [r for r in all_results if 'best_f1' in r]
    for result in sorted(valid_results, key=lambda x: x['best_f1'], reverse=True)[:10]:
        pct_baseline = result['best_f1'] / result['baseline_f1'] * 100 if result.get('baseline_f1') else 0
        print(f"M={result['target_length']}, {result['architecture']}: "
              f"F1={result['best_f1']:.3f} ({pct_baseline:.1f}% of baseline), "
              f"EM={result['final_em']:.3f}, "
              f"Compression={result['compression_ratio']:.1f}x")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SQuAD Compression Ablations")
    parser.add_argument("--target_length", type=int, default=64, help="Target compression length")
    parser.add_argument("--architecture", type=str, default="cross_attention",
                       choices=["cross_attention", "conv", "pooling"])
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--run_all", action="store_true", help="Run all ablations")
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_eval_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    if args.run_all:
        # Run full ablation suite
        run_all_ablations()
    else:
        # Run single configuration
        config = CompressionConfig(
            target_length=args.target_length,
            architecture=args.architecture,
            gpu_id=args.gpu,
            num_train_samples=args.num_train_samples,
            num_eval_samples=args.num_eval_samples,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        result = run_single_ablation(config, args.gpu)

        if result:
            print(f"\nFinal Results:")
            print(f"  Best F1: {result['best_f1']:.4f}")
            print(f"  Final EM: {result['final_em']:.4f}")
            print(f"  Compression: {result['compression_ratio']:.1f}x")