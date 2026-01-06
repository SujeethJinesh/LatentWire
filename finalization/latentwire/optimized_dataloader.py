# latentwire/optimized_dataloader.py
"""
Optimized data loading pipeline for LatentWire training.

This module provides:
1. Multi-worker DataLoader with prefetching
2. Pinned memory for faster GPU transfers
3. Tokenization caching to avoid redundant computation
4. Efficient batching and collation
5. Non-blocking data transfers
"""

import os
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import threading
import queue

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from latentwire.data import load_examples
from latentwire.core_utils import (
    SYSTEM_PROMPT,
    split_user_and_anchor,
    anchor_token_ids,
)


@dataclass
class TokenizedSample:
    """Pre-tokenized sample with all model-specific tokenizations."""
    text: str
    answer: str
    user_text: str  # For chat template mode
    model_tokens: Dict[str, torch.Tensor]  # Model name -> token ids
    model_attention_masks: Dict[str, torch.Tensor]  # Model name -> attention mask


class TokenizationCache:
    """Cache for tokenized samples to avoid redundant tokenization."""

    def __init__(self, cache_dir: str = "runs/token_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # In-memory cache for current session
        self.lock = threading.Lock()

    def _get_cache_key(self, text: str, model_name: str, config: Dict) -> str:
        """Generate unique cache key for tokenization."""
        content = f"{text}_{model_name}_{sorted(config.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model_name: str, config: Dict) -> Optional[torch.Tensor]:
        """Retrieve cached tokenization if available."""
        key = self._get_cache_key(text, model_name, config)

        # Check memory cache first
        with self.lock:
            if key in self.memory_cache:
                return self.memory_cache[key].clone()

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try:
                tokens = torch.load(cache_file, map_location='cpu')
                with self.lock:
                    self.memory_cache[key] = tokens
                return tokens.clone()
            except:
                # Corrupted cache file, remove it
                cache_file.unlink(missing_ok=True)

        return None

    def put(self, text: str, model_name: str, config: Dict, tokens: torch.Tensor):
        """Store tokenization in cache."""
        key = self._get_cache_key(text, model_name, config)

        with self.lock:
            self.memory_cache[key] = tokens.clone()

        # Save to disk asynchronously
        cache_file = self.cache_dir / f"{key}.pt"
        try:
            torch.save(tokens, cache_file)
        except:
            # Non-critical failure, continue without disk cache
            pass


class OptimizedDataset(Dataset):
    """Optimized dataset with pre-tokenization and caching support."""

    def __init__(
        self,
        texts: List[str],
        answers: List[str],
        model_contexts: List[Any],  # List of model contexts from training
        use_chat_template: bool = False,
        strip_anchor_literal: str = "",
        device: str = "cpu",
        cache_tokenization: bool = True,
        prefetch_all: bool = False,
    ):
        self.texts = texts
        self.answers = answers
        self.model_contexts = model_contexts
        self.use_chat_template = use_chat_template
        self.strip_anchor_literal = strip_anchor_literal
        self.device = device
        self.cache_tokenization = cache_tokenization

        # Extract user texts once
        if use_chat_template:
            self.user_texts = [
                split_user_and_anchor(text, strip_anchor_literal)[0]
                for text in texts
            ]
        else:
            self.user_texts = texts

        # Initialize tokenization cache
        if cache_tokenization:
            self.token_cache = TokenizationCache()
        else:
            self.token_cache = None

        # Optionally pre-tokenize everything (for small datasets)
        self.pretokenized = None
        if prefetch_all and len(texts) < 10000:  # Only for small datasets
            print(f"Pre-tokenizing {len(texts)} samples...")
            self.pretokenized = [self._tokenize_sample(i) for i in range(len(texts))]
            print("Pre-tokenization complete.")

    def __len__(self) -> int:
        return len(self.texts)

    def _tokenize_sample(self, idx: int) -> Dict[str, Any]:
        """Tokenize a single sample for all models."""
        text = self.texts[idx]
        answer = self.answers[idx]
        user_text = self.user_texts[idx]

        sample_data = {
            'text': text,
            'answer': answer,
            'user_text': user_text,
            'model_tokens': {},
            'model_attention_masks': {},
        }

        for ctx in self.model_contexts:
            # Check cache first
            cache_key = f"{text}_{ctx.name}_{self.use_chat_template}"
            if self.token_cache:
                cached = self.token_cache.get(text, ctx.name, {'chat': self.use_chat_template})
                if cached is not None:
                    sample_data['model_tokens'][ctx.name] = cached
                    sample_data['model_attention_masks'][ctx.name] = (cached != ctx.wrapper.tokenizer.pad_token_id)
                    continue

            # Tokenize based on mode
            if self.use_chat_template:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
                rendered = ctx.wrapper.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assistant_prefill = ctx.anchor_text if ctx.anchor_mode == "text" else self.strip_anchor_literal
                if assistant_prefill:
                    rendered = rendered + assistant_prefill

                toks = ctx.wrapper.tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )
                tokens = toks["input_ids"][0]
            else:
                anchor_suffix = ctx.anchor_text if ctx.anchor_mode == "text" else self.strip_anchor_literal
                text_with_anchor = f"{text}{anchor_suffix}"
                toks = ctx.wrapper.tokenizer(
                    text_with_anchor,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )
                tokens = toks["input_ids"][0]

            # Cache the tokenization
            if self.token_cache:
                self.token_cache.put(text, ctx.name, {'chat': self.use_chat_template}, tokens)

            sample_data['model_tokens'][ctx.name] = tokens
            sample_data['model_attention_masks'][ctx.name] = torch.ones_like(tokens)

        return sample_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        if self.pretokenized:
            return self.pretokenized[idx]
        return self._tokenize_sample(idx)


def optimized_collate_fn(
    batch: List[Dict[str, Any]],
    model_contexts: List[Any],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Efficient collation function with pinned memory support.

    Returns a dictionary with:
    - texts: List of original texts
    - answers: List of answers
    - user_texts: List of user texts
    - scaffolds: Dict[model_name, padded_token_ids]
    - attention_masks: Dict[model_name, attention_masks]
    """
    texts = [item['text'] for item in batch]
    answers = [item['answer'] for item in batch]
    user_texts = [item['user_text'] for item in batch]

    scaffolds = {}
    attention_masks = {}

    for ctx in model_contexts:
        model_name = ctx.name

        # Collect token sequences for this model
        token_seqs = [item['model_tokens'][model_name] for item in batch]
        attn_seqs = [item['model_attention_masks'][model_name] for item in batch]

        # Determine padding value
        pad_token = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
        if pad_token is None:
            pad_token = int(getattr(ctx.wrapper.tokenizer, "eos_token_id", 0))

        # Pad sequences efficiently
        padded_tokens = pad_sequence(token_seqs, batch_first=True, padding_value=pad_token)
        padded_attn = pad_sequence(attn_seqs, batch_first=True, padding_value=0)

        # Move to device with non-blocking transfer
        scaffolds[model_name] = padded_tokens.to(device, non_blocking=True)
        attention_masks[model_name] = padded_attn.to(device, non_blocking=True)

    return {
        'texts': texts,
        'answers': answers,
        'user_texts': user_texts,
        'scaffolds': scaffolds,
        'attention_masks': attention_masks,
    }


class DataPrefetcher:
    """Prefetch data to GPU while model is computing."""

    def __init__(self, dataloader: DataLoader, device: str = "cuda"):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.preloaded_batch = None
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        """Preload next batch to GPU."""
        try:
            batch = next(self.iter)
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    # Move scaffolds and attention masks to GPU asynchronously
                    for key in ['scaffolds', 'attention_masks']:
                        if key in batch:
                            for model_name in batch[key]:
                                batch[key][model_name] = batch[key][model_name].to(
                                    self.device, non_blocking=True
                                )
            self.preloaded_batch = batch
        except StopIteration:
            self.preloaded_batch = None

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        batch = self.preloaded_batch
        if batch is None:
            raise StopIteration

        # Start loading next batch while current is being processed
        self.preload()

        return batch


def create_optimized_dataloader(
    texts: List[str],
    answers: List[str],
    model_contexts: List[Any],
    batch_size: int = 32,
    num_workers: int = 4,
    use_chat_template: bool = False,
    strip_anchor_literal: str = "",
    device: str = "cuda",
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    cache_tokenization: bool = True,
    use_prefetcher: bool = True,
) -> Union[DataLoader, DataPrefetcher]:
    """
    Create an optimized DataLoader with all performance optimizations.

    Args:
        texts: List of input texts
        answers: List of answers
        model_contexts: List of model contexts from training
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        use_chat_template: Whether to use chat template
        strip_anchor_literal: Anchor literal to strip
        device: Target device
        shuffle: Whether to shuffle data
        pin_memory: Use pinned memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        cache_tokenization: Cache tokenizations to disk
        use_prefetcher: Use GPU prefetcher for async transfers

    Returns:
        DataLoader or DataPrefetcher instance
    """

    # Determine optimal number of workers
    if num_workers == -1:
        # Auto-detect based on CPU count and dataset size
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if len(texts) < 1000:
            num_workers = 0  # Small dataset, use main process
        else:
            num_workers = min(cpu_count // 2, 8)  # Use half CPUs, max 8

    # Create dataset
    dataset = OptimizedDataset(
        texts=texts,
        answers=answers,
        model_contexts=model_contexts,
        use_chat_template=use_chat_template,
        strip_anchor_literal=strip_anchor_literal,
        device="cpu",  # Keep data on CPU until collation
        cache_tokenization=cache_tokenization,
        prefetch_all=(len(texts) < 5000),  # Pre-tokenize small datasets
    )

    # Create collate function with model contexts bound
    def collate_fn(batch):
        return optimized_collate_fn(batch, model_contexts, device)

    # Create DataLoader with optimizations
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )

    # Optionally wrap with GPU prefetcher
    if use_prefetcher and torch.cuda.is_available():
        return DataPrefetcher(dataloader, device)

    return dataloader


def benchmark_dataloader(
    dataloader: Union[DataLoader, DataPrefetcher],
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> Dict[str, float]:
    """
    Benchmark dataloader performance.

    Returns:
        Dictionary with timing statistics
    """
    print(f"Benchmarking dataloader ({num_batches} batches)...")

    timings = []

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        # Ensure GPU synchronization for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Actual benchmark
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start = time.time()

        # Simulate minimal processing to measure pure loading time
        scaffolds = batch.get('scaffolds', {})
        for model_name, tokens in scaffolds.items():
            _ = tokens.shape  # Access tensor to ensure transfer is complete

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        timings.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(timings[-10:])
            print(f"  Batch {i+1}/{num_batches}: {avg_time*1000:.2f}ms/batch")

    results = {
        'mean_time_ms': np.mean(timings) * 1000,
        'std_time_ms': np.std(timings) * 1000,
        'min_time_ms': np.min(timings) * 1000,
        'max_time_ms': np.max(timings) * 1000,
        'batches_per_second': 1.0 / np.mean(timings),
    }

    print("\nDataLoader Benchmark Results:")
    print(f"  Mean time: {results['mean_time_ms']:.2f}ms")
    print(f"  Std dev: {results['std_time_ms']:.2f}ms")
    print(f"  Min/Max: {results['min_time_ms']:.2f}ms / {results['max_time_ms']:.2f}ms")
    print(f"  Throughput: {results['batches_per_second']:.1f} batches/sec")

    return results