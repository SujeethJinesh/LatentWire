#!/usr/bin/env python3
"""
Stage 1 Phase 1: Pure Reconstruction Training
==============================================

Trains an adapter to reconstruct original embeddings from PCA-compressed ones.

Architecture:
    Text → LLM Embeddings [seq_len, 4096]
         → PCA Compression [seq_len, compress_dim]
         → Learned Adapter [seq_len, 4096]
         → LLM Generation

Loss: Cosine Similarity (direction) + MSE (magnitude)

Hypothesis: Good reconstruction → Good generation
If adapter can perfectly reconstruct embeddings, LLM should generate correctly.

Usage:
    python train_adapter_only_phase1.py \
        --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --compress_dim 1024 \
        --compress_method pca \
        --samples 10000 \
        --epochs 3 \
        --batch_size 32
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

# Add latentwire to path
sys.path.insert(0, str(Path(__file__).parent))

from latentwire.data import load_examples
from latentwire.models import Adapter
from latentwire.core_utils import batch_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM


class EmbeddingCompressor:
    """PCA-based embedding compressor."""

    def __init__(self, input_dim: int, output_dim: int, method: str = "pca"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.projection: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.explained_variance_ratio: Optional[float] = None

    def initialize_from_pca(
        self,
        *,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance_ratio: np.ndarray,
    ) -> None:
        """Populate internal state from fitted PCA/IncrementalPCA outputs."""
        self.projection = torch.from_numpy(components.T).float()
        self.mean = torch.from_numpy(mean).float()
        self.explained_variance_ratio = float(explained_variance_ratio.sum())

    def fit(self, embeddings: torch.Tensor):
        """Fit PCA on embeddings.

        Args:
            embeddings: [num_vectors, input_dim] tensor
        """
        if self.method != "pca":
            raise ValueError(f"Unsupported method: {self.method}")

        # Convert to numpy
        embeddings_np = embeddings.cpu().float().numpy()

        # Fit PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.output_dim)
        pca.fit(embeddings_np)

        self.initialize_from_pca(
            components=pca.components_,
            mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress embeddings using fitted PCA.

        Args:
            embeddings: [batch, seq_len, input_dim] tensor

        Returns:
            compressed: [batch, seq_len, output_dim] tensor
        """
        if self.projection is None:
            # Fallback: truncate to output_dim
            return embeddings[..., :self.output_dim]

        orig_dtype = embeddings.dtype
        device = embeddings.device

        # Move projection to same device
        projection = self.projection.to(device)
        mean = self.mean.to(device)

        # Center and project: (x - mean) @ projection
        shape = embeddings.shape
        embeddings_flat = embeddings.view(-1, self.input_dim).float()  # [batch*seq, input_dim]
        centered = embeddings_flat - mean  # [batch*seq, input_dim]
        compressed = centered @ projection  # [batch*seq, output_dim]
        compressed = compressed.view(*shape[:-1], self.output_dim)  # [batch, seq, output_dim]

        return compressed.to(orig_dtype)


def compute_reconstruction_metrics(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor
) -> Dict[str, float]:
    """Compute reconstruction quality metrics.

    Args:
        reconstructed: [batch, seq, dim]
        original: [batch, seq, dim]
        mask: [batch, seq] attention mask

    Returns:
        Dictionary of metrics
    """
    # Convert to float32 for stable computation
    reconstructed = reconstructed.float()
    original = original.float()
    mask = mask.float().unsqueeze(-1)  # [batch, seq, 1]

    # Apply mask
    reconstructed_masked = reconstructed * mask
    original_masked = original * mask

    # MSE
    mse = F.mse_loss(reconstructed_masked, original_masked)

    # Cosine similarity (per-token, then averaged)
    cos_sim = F.cosine_similarity(
        reconstructed.view(-1, reconstructed.shape[-1]),
        original.view(-1, original.shape[-1]),
        dim=-1
    )
    mask_flat = mask.view(-1)
    cos_sim_masked = (cos_sim * mask_flat).sum() / (mask_flat.sum() + 1e-8)

    # Relative error: ||recon - orig|| / ||orig||
    diff_norm = (reconstructed_masked - original_masked).norm()
    orig_norm = original_masked.norm()
    rel_error = diff_norm / (orig_norm + 1e-8)

    return {
        "recon_mse": mse.item(),
        "recon_cosine_sim": cos_sim_masked.item(),
        "recon_rel_error": rel_error.item(),
    }


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


class EmbeddingDataset(Dataset):
    """Dataset that yields (source_text, answer_text, source_embeds)."""

    def __init__(self, examples: List[Dict], tokenizer, model, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

        # Get embedding device
        self.embed_device = model.get_input_embeddings().weight.device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Tokenize source
        encoded = self.tokenizer(
            ex['source'],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        input_ids = encoded['input_ids'][0]  # [seq_len]

        # Extract embeddings
        with torch.no_grad():
            input_ids_device = input_ids.to(self.embed_device)
            embeddings = self.model.get_input_embeddings()(input_ids_device)  # [seq_len, d_model]
            embeddings = embeddings.cpu()

        return {
            'source': ex['source'],
            'answer': ex['answer'],
            'embeddings': embeddings,  # [seq_len, d_model]
            'input_ids': input_ids,
        }


def collate_fn(batch, pad_token_id: int):
    """Collate batch with padding."""
    # Find max length
    max_len = max(item['embeddings'].shape[0] for item in batch)

    batch_size = len(batch)
    d_model = batch[0]['embeddings'].shape[1]

    # Initialize tensors
    embeddings = torch.zeros(batch_size, max_len, d_model)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len)

    sources = []
    answers = []

    for i, item in enumerate(batch):
        seq_len = item['embeddings'].shape[0]
        embeddings[i, :seq_len] = item['embeddings']
        input_ids[i, :seq_len] = item['input_ids']
        mask[i, :seq_len] = 1
        sources.append(item['source'])
        answers.append(item['answer'])

    return {
        'embeddings': embeddings,  # [batch, seq, d_model]
        'input_ids': input_ids,  # [batch, seq]
        'mask': mask,  # [batch, seq]
        'sources': sources,
        'answers': answers,
    }


def train_epoch(
    adapter: nn.Module,
    compressor: EmbeddingCompressor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    adapter_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    diagnostic_log: str,
    *,
    model: Optional[nn.Module],
    model_device: Optional[torch.device],
    model_dtype: Optional[torch.dtype],
    gen_loss_weight: float,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.1,
) -> int:
    """Train for one epoch."""
    adapter.train()

    total_loss = 0.0
    total_cosine_sim = 0.0
    total_mse = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        embeddings = batch['embeddings'].to(device, dtype=adapter_dtype)  # [batch, seq, d_model]
        mask = batch['mask'].to(device)  # [batch, seq]

        # Compress embeddings
        compressed = compressor.compress(embeddings)  # [batch, seq, compress_dim]
        compressed = compressed.to(device, dtype=adapter_dtype)

        # Reconstruct through adapter
        reconstructed = adapter(compressed)  # [batch, seq, d_model]

        # Compute reconstruction loss
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq, 1]

        # MSE loss
        mse = F.mse_loss(reconstructed * mask_expanded, embeddings * mask_expanded)

        # Cosine similarity loss (maximize similarity = minimize 1 - similarity)
        cos_sim = F.cosine_similarity(
            reconstructed.view(-1, reconstructed.shape[-1]),
            embeddings.view(-1, embeddings.shape[-1]),
            dim=-1
        )
        mask_flat = mask.view(-1)
        cos_sim_mean = (cos_sim * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        cosine_loss = 1.0 - cos_sim_mean

        # Combined loss
        loss = cosine_weight * cosine_loss + mse_weight * mse

        gen_loss_val = 0.0
        if gen_loss_weight > 0.0:
            assert model is not None and model_device is not None and model_dtype is not None, \
                "Model context required when gen_loss_weight > 0"

            labels = batch['input_ids'].to(model_device).clone()
            mask_model = batch['mask'].to(model_device)
            labels[mask_model == 0] = -100

            reconstructed_for_model = reconstructed.to(model_device, dtype=model_dtype)
            model_outputs = model(
                inputs_embeds=reconstructed_for_model,
                labels=labels,
            )
            gen_loss = model_outputs.loss
            gen_loss_val = gen_loss.item()

            loss = loss + gen_loss_weight * gen_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_cosine_sim += cos_sim_mean.item()
        total_mse += mse.item()

        global_step += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cos': f'{cos_sim_mean.item():.3f}',
            'mse': f'{mse.item():.4f}',
            'gen': f'{gen_loss_val:.4f}' if gen_loss_weight > 0 else 'n/a',
        })

        # Log every 50 steps
        if global_step % 50 == 0:
            log_diagnostics(diagnostic_log, global_step, epoch, {
                'type': 'train_step',
                'loss': loss.item(),
                'cosine_loss': cosine_loss.item(),
                'mse': mse.item(),
                'cosine_sim': cos_sim_mean.item(),
                'gen_loss': gen_loss_val if gen_loss_weight > 0 else None,
            })

    # Log epoch summary
    num_batches = len(dataloader)
    log_diagnostics(diagnostic_log, global_step, epoch, {
        'type': 'train_epoch',
        'avg_loss': total_loss / num_batches,
        'avg_cosine_sim': total_cosine_sim / num_batches,
        'avg_mse': total_mse / num_batches,
        'gen_loss_weight': gen_loss_weight,
    })

    return global_step


@torch.no_grad()
def evaluate(
    adapter: nn.Module,
    compressor: EmbeddingCompressor,
    model: nn.Module,
    tokenizer,
    dataset: List[Dict],
    *,
    adapter_dtype: torch.dtype,
    max_new_tokens: int = 12,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Evaluate adapter on generation task."""
    adapter.eval()
    model.eval()

    predictions = []
    references = []

    embed_device = model.get_input_embeddings().weight.device
    adapter_device = next(adapter.parameters()).device
    model_dtype = model.get_input_embeddings().weight.dtype

    print(f"\nEvaluating on {len(dataset)} examples...")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Eval"):
        batch = dataset[i:i+batch_size]

        # Tokenize batch
        sources = [ex['source'] for ex in batch]
        encoded = tokenizer(
            sources,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )

        input_ids = encoded['input_ids'].to(embed_device)

        # Extract embeddings
        text_embeds = model.get_input_embeddings()(input_ids)  # [batch, seq, d_model]

        # Compress and reconstruct
        compressed = compressor.compress(text_embeds)  # [batch, seq, compress_dim]
        compressed = compressed.to(adapter_device, dtype=adapter_dtype)
        reconstructed = adapter(compressed)  # [batch, seq, d_model]
        reconstructed = reconstructed.to(embed_device, dtype=model_dtype)

        # Generate
        outputs = model.generate(
            inputs_embeds=reconstructed,
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
        'exact_match': em,
        'f1_score': f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pure reconstruction training")

    # Model
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')

    # Compression
    parser.add_argument('--compress_dim', type=int, default=1024, help='PCA output dimension')
    parser.add_argument('--compress_method', type=str, default='pca', choices=['pca'])
    parser.add_argument('--pca_samples', type=int, default=5000, help='Samples for PCA fitting')
    parser.add_argument('--pca_batch_size', type=int, default=512, help='Batch size (in examples) when extracting embeddings for PCA')
    parser.add_argument('--pca_token_cap', type=int, default=None, help='Optional cap on total tokens used for PCA (useful to bound memory/cost)')
    parser.add_argument('--pca_tokens_per_example', type=int, default=0,
                        help='Maximum tokens sampled per example when fitting PCA (0 or negative = use all tokens).')
    parser.add_argument('--pca_flush_tokens', type=int, default=8192,
                        help='[DEPRECATED] No longer used with standard PCA.')
    parser.add_argument('--pca_cache_path', type=str, default=None,
                        help='Optional path to load/save PCA components (speeds up reruns).')

    # Adapter
    parser.add_argument('--adapter_hidden_mult', type=int, default=4)
    parser.add_argument('--adapter_dropout', type=float, default=0.1)
    parser.add_argument('--adapter_lr', type=float, default=5e-4)

    # Training
    parser.add_argument('--dataset', type=str, default='squad', choices=['squad', 'hotpot', 'squad_v2'])
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)

    # Loss weights
    parser.add_argument('--cosine_weight', type=float, default=1.0)
    parser.add_argument('--mse_weight', type=float, default=0.1)

    # Evaluation
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=12)

    # Optional generation/LoRA enhancement
    parser.add_argument('--gen_loss_weight', type=float, default=0.0,
                        help='Weight for teacher-forced generation loss (enables gradients through LLM).')
    parser.add_argument('--use_lora', action='store_true', help='Apply LoRA adapters to the LLM (requires gen_loss_weight > 0).')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_layers', type=int, default=None, help='Apply LoRA to first N layers (default all).')

    # I/O
    parser.add_argument('--save_dir', type=str, default='./runs/phase1')
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

    print(f"GPUs available: {torch.cuda.device_count()}\n")

    # Load model
    print(f"Loading {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    embed_device = model.get_input_embeddings().weight.device
    model_dtype = model.get_input_embeddings().weight.dtype

    if args.gen_loss_weight > 0 and not args.use_lora:
        warnings.warn(
            "Generation loss enabled without LoRA: only the adapter will be updated via generation gradients.",
            RuntimeWarning,
        )
    if args.use_lora and args.gen_loss_weight <= 0:
        raise ValueError("LoRA requires --gen_loss_weight > 0 to supply gradients through the LLM.")

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        print(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=list(range(args.lora_layers)) if args.lora_layers else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print()
        # Refresh embedding device/dtype in case PEFT wrapping moved tensors
        embed_device = model.get_input_embeddings().weight.device
        model_dtype = model.get_input_embeddings().weight.dtype

    if args.gen_loss_weight > 0 or args.use_lora:
        model.train()
    else:
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print(f"Model loaded!\n")

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

    # Fit PCA
    tokens_per_example = args.pca_tokens_per_example
    if tokens_per_example is not None and tokens_per_example <= 0:
        tokens_per_example = None

    print("Fitting PCA on {} examples (batch={}, tokens/example={})...".format(
        args.pca_samples,
        args.pca_batch_size,
        tokens_per_example if tokens_per_example is not None else "all"
    ))
    d_model = model.config.hidden_size

    compressor = EmbeddingCompressor(
        input_dim=d_model,
        output_dim=args.compress_dim,
        method=args.compress_method
    )

    pca_cache_path = Path(args.pca_cache_path) if args.pca_cache_path else None

    embed_device = model.get_input_embeddings().weight.device

    if pca_cache_path and pca_cache_path.exists():
        try:
            state = torch.load(pca_cache_path, map_location="cpu")
            cache_meta = state.get("meta", {}) if isinstance(state, dict) else {}
            cached_tokens = cache_meta.get("tokens_per_example")
            if (tokens_per_example is None and cached_tokens not in (None, 0)) or (
                tokens_per_example is not None and cached_tokens != tokens_per_example
            ):
                raise ValueError(
                    f"Cached PCA uses tokens_per_example={cached_tokens}, but current run requests {tokens_per_example}."
                )
            compressor.initialize_from_pca(
                components=state["components"],
                mean=state["mean"],
                explained_variance_ratio=state["explained_variance_ratio"],
            )
            print(f"Loaded PCA cache from {pca_cache_path}. Explained variance: {compressor.explained_variance_ratio:.2%}\n")
            cache_loaded = True
        except Exception as exc:
            print(f"[WARN] Failed to load PCA cache {pca_cache_path}: {exc}. Recomputing.")
            pca_cache_path.unlink(missing_ok=True)
            cache_loaded = False
    else:
        cache_loaded = False
    if not cache_loaded:
        # Collect embeddings for PCA fitting (batched for GPU efficiency)
        pca_examples = train_examples[:args.pca_samples]

        all_embeddings = []
        total_tokens = 0
        token_cap = args.pca_token_cap

        for i in tqdm(range(0, len(pca_examples), args.pca_batch_size), desc="Extracting embeddings for PCA"):
            batch = pca_examples[i:i+args.pca_batch_size]
            if not batch:
                break
            sources = [ex['source'] for ex in batch]

            encoded = tokenizer(
                sources,
                return_tensors='pt',
                truncation=True,
                max_length=args.max_length,
                padding=True
            )

            input_ids = encoded['input_ids'].to(embed_device)

            with torch.no_grad():
                embeddings = model.get_input_embeddings()(input_ids)  # [batch, seq, d_model]

            # Extract full sequences (not random tokens)
            for j in range(embeddings.shape[0]):
                seq_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()
                if token_cap is not None and total_tokens >= token_cap:
                    break
                if token_cap is not None and total_tokens + seq_len > token_cap:
                    seq_len = token_cap - total_tokens
                if seq_len <= 0:
                    break

                # Use full sequence (keep all tokens for exact PCA)
                token_slice = embeddings[j, :seq_len]
                if tokens_per_example is not None and seq_len > tokens_per_example:
                    token_slice = token_slice[:tokens_per_example]
                    seq_len = token_slice.shape[0]

                all_embeddings.append(token_slice.cpu().float())
                total_tokens += seq_len

            if token_cap is not None and total_tokens >= token_cap:
                break

        if len(all_embeddings) == 0:
            raise RuntimeError(
                "No embeddings collected for PCA. "
                "Increase --pca_samples or --pca_batch_size."
            )

        # Concatenate all embeddings and fit standard PCA
        all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [total_tokens, d_model]
        print(f"Fitting standard PCA on {all_embeddings_tensor.shape[0]} token embeddings...")

        # Use standard PCA (exact eigendecomposition)
        pca = PCA(n_components=args.compress_dim)
        pca.fit(all_embeddings_tensor.numpy())

        compressor.initialize_from_pca(
            components=pca.components_,
            mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        print(f"PCA fitted: {d_model}D → {args.compress_dim}D")
        print(f"Explained variance: {compressor.explained_variance_ratio:.2%}\n")

        if pca_cache_path:
            pca_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "components": pca.components_,
                    "mean": pca.mean_,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "meta": {
                        "tokens_per_example": tokens_per_example,
                        "total_tokens": total_tokens,
                        "algorithm": "standard_pca",  # Mark as standard PCA
                    },
                },
                pca_cache_path,
            )
            print(f"Saved PCA cache to {pca_cache_path}\n")

    # Create adapter
    print(f"Creating adapter: {args.compress_dim}D → {d_model}D")
    adapter = Adapter(
        d_z=args.compress_dim,
        d_model=d_model,
        latent_length=32,  # Not used in this mode, but required by constructor
        hidden_mult=args.adapter_hidden_mult,
        dropout=args.adapter_dropout,
        enable_metadata=False,
        colorize=False,
    ).to(embed_device)

    num_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Adapter parameters: {num_params:,}\n")

    # Optimizer
    trainable_params = list(adapter.parameters())
    if args.use_lora:
        trainable_params += [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.adapter_lr)

    # Create dataset and dataloader
    print("Creating training dataloader...")
    train_dataset = EmbeddingDataset(train_examples, tokenizer, model, args.max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0,  # Use 0 for compatibility
    )
    print(f"Training batches: {len(train_dataloader)}\n")

    # Training loop
    print("="*80)
    print("TRAINING")
    print("="*80)
    print()

    best_f1 = 0.0
    global_step = 0
    adapter_dtype = next(adapter.parameters()).dtype

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}\n")

        # Train
        use_model_grad = args.gen_loss_weight > 0.0
        global_step = train_epoch(
            adapter=adapter,
            compressor=compressor,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=embed_device,
            adapter_dtype=adapter_dtype,
            epoch=epoch,
            global_step=global_step,
            diagnostic_log=args.diagnostic_log,
            model=model if use_model_grad else None,
            model_device=embed_device if use_model_grad else None,
            model_dtype=model_dtype if use_model_grad else None,
            gen_loss_weight=args.gen_loss_weight,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
        )

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nEvaluating epoch {epoch + 1}...")
            eval_metrics = evaluate(
                adapter=adapter,
                compressor=compressor,
                model=model,
                tokenizer=tokenizer,
                dataset=eval_examples,
                adapter_dtype=adapter_dtype,
                max_new_tokens=args.max_new_tokens,
                batch_size=8,
            )

            print(f"\nEval Results:")
            print(f"  EM: {eval_metrics['em']:.2%}")
            print(f"  F1: {eval_metrics['f1']:.2%}")

            # Log evaluation
            log_diagnostics(args.diagnostic_log, global_step, epoch, {
                'type': 'full_eval',
                'em': eval_metrics['em'],
                'f1': eval_metrics['f1'],
            })

            # Save best checkpoint
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']

                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'adapter_state_dict': adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'config': {
                        'input_dim': d_model,
                        'compress_dim': args.compress_dim,
                        'compress_method': args.compress_method,
                        'adapter_hidden_mult': args.adapter_hidden_mult,
                        'samples': args.samples,
                        'pca_explained_variance': compressor.explained_variance_ratio,
                    },
                    'compressor': {
                        'projection': compressor.projection,
                        'mean': compressor.mean,
                        'explained_variance_ratio': compressor.explained_variance_ratio,
                    }
                }

                if args.use_lora:
                    checkpoint['lora_state_dict'] = model.state_dict()

                ckpt_path = save_dir / 'adapter_phase1_best.pt'
                torch.save(checkpoint, ckpt_path)
                print(f"  → Saved best checkpoint: {ckpt_path}")

            if args.gen_loss_weight > 0 or args.use_lora:
                model.train()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest F1: {best_f1:.2%}")
    print(f"Results saved to: {save_dir}")
    print()


if __name__ == '__main__':
    main()
