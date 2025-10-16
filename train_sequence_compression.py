#!/usr/bin/env python3
"""
Experiment 1: Moderate Sequence Compression with LoRA

Compresses sequence length (e.g., 300 → 128) while preserving positional information
and using LoRA to help the frozen LLM adapt to pooled representations.

Architecture:
    Text → Embeddings [seq, 4096] → Learned Pooling [target_seq, 4096] → LoRA-LLM → Answer

Key innovations:
- Positional encoding preservation during pooling
- LoRA adaptation for reading compressed sequences
- Multiple pooling strategies with position awareness
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
    """Learned cross-attention pooling with positional awareness.

    Compresses sequence via learned queries that attend to input tokens
    while preserving positional information.
    """

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


class ConvolutionalPooling(nn.Module):
    """Strided convolution pooling with positional preservation."""

    def __init__(self, input_dim: int, target_length: int, source_length: int):
        super().__init__()
        self.target_length = target_length
        self.stride = source_length // target_length

        # 1D conv with learned kernels
        self.conv = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=self.stride,
            stride=self.stride,
            groups=input_dim  # Depthwise
        )
        self.norm = nn.LayerNorm(input_dim)
        self.pos_encoding = PositionalEncoding(input_dim)

    def forward(self, x: torch.Tensor, src_positions: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq, dim]"""
        batch_size, seq_len, dim = x.shape

        # Transpose for conv1d: [batch, dim, seq]
        x_t = x.transpose(1, 2)

        # Convolve
        pooled = self.conv(x_t)  # [batch, dim, target_len]
        pooled = pooled.transpose(1, 2)  # [batch, target_len, dim]

        # Add positional encoding at output positions
        target_positions = torch.arange(0, self.target_length, device=x.device)
        target_positions = target_positions.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoding(target_positions)

        pooled = pooled + pos_emb
        pooled = self.norm(pooled)

        return pooled


class SequenceCompressor(nn.Module):
    """Main module for sequence compression."""

    def __init__(
        self,
        input_dim: int,
        target_length: int,
        pooling_method: str = "learned_attention",
        source_length: int = 300,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_length = target_length
        self.pooling_method = pooling_method

        if pooling_method == "learned_attention":
            self.pooler = LearnedAttentionPooling(input_dim, target_length)
        elif pooling_method == "convolutional":
            self.pooler = ConvolutionalPooling(input_dim, target_length, source_length)
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
) -> int:
    """Train for one epoch."""
    compressor.train()
    if hasattr(model, 'enable_adapter_layers'):
        model.train()  # LoRA layers
    else:
        model.eval()  # Frozen base

    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        source_ids = batch['source_ids'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        src_mask = batch['src_mask'].to(device)
        ans_mask = batch['ans_mask'].to(device)

        # Get source embeddings
        with torch.no_grad():
            source_embeds = model.get_input_embeddings()(source_ids)  # [batch, src_seq, dim]

        # Create position tensor
        batch_size, src_seq = source_ids.shape
        positions = torch.arange(src_seq, device=device).unsqueeze(0).expand(batch_size, -1)

        # Compress sequence
        compressed = compressor(source_embeds, positions)  # [batch, target_len, dim]

        # Get answer embeddings (teacher-forced)
        answer_embeds = model.get_input_embeddings()(answer_ids[:, :-1])  # [batch, ans_seq-1, dim]

        # Concatenate compressed prefix with answer embeddings
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
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
        if hasattr(model, 'enable_adapter_layers'):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log every 50 steps
        if global_step % 50 == 0:
            log_diagnostics(diagnostic_log, global_step, epoch, {
                'type': 'train_step',
                'loss': loss.item(),
            })

    # Log epoch summary
    num_batches = len(dataloader)
    log_diagnostics(diagnostic_log, global_step, epoch, {
        'type': 'train_epoch',
        'avg_loss': total_loss / num_batches,
    })

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
) -> Dict[str, float]:
    """Evaluate compressed sequence generation."""
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

    return {
        'em': em,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Sequence compression with LoRA")

    # Model
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')

    # Compression
    parser.add_argument('--target_sequence_length', type=int, default=128,
                       help='Target compressed sequence length')
    parser.add_argument('--source_length', type=int, default=300,
                       help='Expected source sequence length')
    parser.add_argument('--pooling_method', type=str, default='learned_attention',
                       choices=['learned_attention', 'convolutional'])

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

    # Force CUDA device and handle MPS issues on clusters
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # Bypass broken MPS daemon entirely (cluster issue)
    # MPS Error 805 means the daemon is in a bad state
    # Solution: disable MPS by setting pipes to /dev/null
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/dev/null'
    os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/dev/null'

    # Try to initialize CUDA
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
        print("Trying to reset CUDA...")
        # Clear CUDA cache and retry
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # Kill MPS if it exists
        os.system('echo quit | nvidia-cuda-mps-control 2>/dev/null')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            raise RuntimeError("Cannot initialize CUDA. GPUs visible but not accessible. "
                             "Try: export CUDA_VISIBLE_DEVICES=0,1,2,3 && "
                             "echo quit | nvidia-cuda-mps-control")

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
        source_length=args.source_length,
    ).to(embed_device).to(model_dtype)  # Match model's dtype (bfloat16)

    num_params = sum(p.numel() for p in compressor.parameters())
    print(f"Compressor parameters: {num_params:,}\n")

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
            )

            print(f"\nEval Results:")
            print(f"  EM: {eval_metrics['em']:.2%}")
            print(f"  F1: {eval_metrics['f1']:.2%}")

            # Log
            log_diagnostics(args.diagnostic_log, global_step, epoch, {
                'type': 'full_eval',
                'em': eval_metrics['em'],
                'f1': eval_metrics['f1'],
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
