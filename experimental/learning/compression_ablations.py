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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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
    architecture: str = "cross_attention"  # cross_attention, conv, pooling

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
            self.loss_weights = {
                'teacher_forcing': 0.5,
                'kl_distill': 0.3,
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

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize answer
        answer_tokens = self.tokenizer(
            item['answer'],
            max_length=32,  # Answers are typically short
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': prompt_tokens['input_ids'].squeeze(0),
            'attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'answer_ids': answer_tokens['input_ids'].squeeze(0),
            'answer_mask': answer_tokens['attention_mask'].squeeze(0),
            'answer_text': item['answer']
        }


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
            'eval_f1': [],
            'eval_em': [],
            'compression_ratio': config.max_input_length / config.target_length
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

        # Apply LoRA
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

        # Initialize compressor
        hidden_dim = self.base_model.config.hidden_size

        if self.config.architecture == "cross_attention":
            self.compressor = CrossAttentionCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device)
        elif self.config.architecture == "conv":
            self.compressor = ConvolutionalCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device)
        elif self.config.architecture == "pooling":
            self.compressor = WeightedPoolingCompressor(
                target_length=self.config.target_length,
                hidden_dim=hidden_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

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

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size * 2,  # Can use larger batch for eval
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def compute_loss(self, batch):
        """Compute training loss with multiple components."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        answer_ids = batch['answer_ids'].to(self.device)
        answer_mask = batch['answer_mask'].to(self.device)

        # Get full embeddings
        with torch.no_grad():
            full_embeds = self.embed_layer(input_ids)

        # Compress
        compressed, attn_weights = self.compressor(full_embeds)

        # Forward pass through model with compressed inputs
        outputs = self.model(inputs_embeds=compressed)
        student_logits = outputs.logits

        losses = {}

        # 1. Teacher-forcing loss on answer generation
        if self.config.loss_weights['teacher_forcing'] > 0:
            # Generate answer autoregressively with teacher forcing
            # Append answer tokens to compressed sequence
            answer_embeds = self.embed_layer(answer_ids)
            combined = torch.cat([compressed, answer_embeds], dim=1)

            # Forward pass with combined sequence
            combined_outputs = self.model(inputs_embeds=combined)
            combined_logits = combined_outputs.logits

            # Compute loss on answer portion
            answer_start = compressed.size(1)
            answer_logits = combined_logits[:, answer_start-1:-1, :]

            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            teacher_forcing_loss = loss_fct(
                answer_logits.reshape(-1, answer_logits.size(-1)),
                answer_ids.reshape(-1)
            )
            losses['teacher_forcing'] = teacher_forcing_loss

        # 2. KL distillation from full model
        if self.config.loss_weights['kl_distill'] > 0:
            with torch.no_grad():
                # Get teacher outputs with full input
                teacher_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            # KL divergence on overlapping positions
            min_len = min(student_logits.size(1), teacher_logits.size(1))
            kl_loss = F.kl_div(
                F.log_softmax(student_logits[:, :min_len, :] / 3.0, dim=-1),
                F.softmax(teacher_logits[:, :min_len, :] / 3.0, dim=-1).detach(),
                reduction='batchmean'
            )
            losses['kl_distill'] = kl_loss

        # 3. Contrastive loss to prevent collapse
        if self.config.loss_weights['contrastive'] > 0 and compressed.size(0) > 1:
            # Normalize compressed representations
            compressed_norm = F.normalize(compressed.mean(dim=1), p=2, dim=-1)

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

        # Combine losses
        total_loss = sum(
            self.config.loss_weights[k] * v
            for k, v in losses.items()
            if k in self.config.loss_weights
        )

        return total_loss, losses

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.compressor.train()

        epoch_losses = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Compute loss
            loss, loss_dict = self.compute_loss(batch)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation
            loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track metrics
            epoch_losses.append(loss.item() * self.config.gradient_accumulation)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{epoch_losses[-1]:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        return np.mean(epoch_losses)

    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        self.compressor.eval()

        all_predictions = []
        all_references = []

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

        # Compute F1 and EM scores
        f1_scores = []
        em_scores = []

        for pred, ref in zip(all_predictions, all_references):
            # Simple F1 calculation
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            common = set(pred_tokens) & set(ref_tokens)

            if len(common) == 0:
                f1 = 0.0
            else:
                precision = len(common) / len(pred_tokens) if pred_tokens else 0
                recall = len(common) / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

            # Exact match
            em = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
            em_scores.append(em)

        return np.mean(f1_scores), np.mean(em_scores)

    def train(self):
        """Main training loop."""
        # Setup
        self.setup_model()
        self.setup_data()

        # Initialize optimizer
        optimizer_groups = [
            {'params': self.compressor.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.parameters(), 'lr': self.config.lora_lr}
        ]
        self.optimizer = AdamW(optimizer_groups, weight_decay=0.01)

        # Learning rate scheduler with warmup
        num_training_steps = len(self.train_loader) * self.config.epochs // self.config.gradient_accumulation
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=num_training_steps - num_warmup_steps,
            T_mult=1
        )

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

            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Eval F1: {f1_score:.4f}")
            print(f"  Eval EM: {em_score:.4f}")

            # Save best model
            if f1_score > best_f1:
                best_f1 = f1_score
                self.save_checkpoint(epoch, is_best=True)

            # Save metrics
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

def run_single_ablation(config: CompressionConfig, gpu_id: int):
    """Run a single ablation on specified GPU."""
    config.gpu_id = gpu_id

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"\n{'='*80}")
    print(f"Starting ablation: M={config.target_length}, {config.architecture}")
    print(f"GPU: {gpu_id}, Loss weights: {config.loss_weights}")
    print(f"{'='*80}\n")

    try:
        trainer = CompressionTrainer(config)
        trainer.train()

        # Return final metrics
        return {
            'config': config,
            'final_f1': trainer.metrics['eval_f1'][-1],
            'final_em': trainer.metrics['eval_em'][-1],
            'best_f1': max(trainer.metrics['eval_f1']),
            'compression_ratio': trainer.metrics['compression_ratio']
        }
    except Exception as e:
        print(f"Error in ablation: {e}")
        return None


def run_all_ablations():
    """Run all ablations across 4 GPUs."""

    # Define ablation grid
    ablations = []

    # Compression ratios
    target_lengths = [32, 64, 128]

    # Architectures
    architectures = ["cross_attention", "conv", "pooling"]

    # Loss weight configurations
    loss_configs = [
        {'teacher_forcing': 0.7, 'kl_distill': 0.2, 'contrastive': 0.1},  # Task-focused
        {'teacher_forcing': 0.4, 'kl_distill': 0.4, 'contrastive': 0.2},  # Balanced
        {'teacher_forcing': 0.2, 'kl_distill': 0.6, 'contrastive': 0.2},  # Distillation-focused
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
    results = []

    for i in range(0, len(ablations), 4):
        batch = ablations[i:i+4]
        processes = []

        # Start processes for this batch
        for j, config in enumerate(batch):
            p = mp.Process(
                target=lambda c, g: results.append(run_single_ablation(c, g)),
                args=(config, j % 4)
            )
            p.start()
            processes.append(p)

        # Wait for batch to complete
        for p in processes:
            p.join()

    # Save all results
    results_file = Path("runs/compression_ablations/all_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll ablations complete! Results saved to {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)

    for result in sorted(results, key=lambda x: x['best_f1'] if x else 0, reverse=True)[:10]:
        if result:
            print(f"M={result['config'].target_length}, {result['config'].architecture}: "
                  f"F1={result['best_f1']:.3f}, EM={result['final_em']:.3f}, "
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