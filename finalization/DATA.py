#!/usr/bin/env python3
"""
================================================================================
DATA.py - Data Module for LatentWire/Telepathy
================================================================================

This module contains all data-related components including datasets,
data loaders, collators, and data processing utilities.

Author: LatentWire Team
Date: January 2025
Version: 1.0.0 (Split from consolidated)
================================================================================
"""

import random
from typing import Dict, List, Any, Optional
from pathlib import Path

# Scientific computing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Deep learning
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "squad"
    max_samples: int = -1  # -1 for all
    max_prefix_len: int = 512
    max_answer_len: int = 128

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Augmentation
    augment: bool = False
    noise_prob: float = 0.1

    # Compression settings (for compatibility)
    compression: Optional[Any] = None
    training: Optional[Any] = None

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "max_samples": self.max_samples,
            "max_prefix_len": self.max_prefix_len,
            "max_answer_len": self.max_answer_len,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "augment": self.augment,
            "noise_prob": self.noise_prob,
        }


# ============================================================================
# BASE DATASET
# ============================================================================

class BaseDataset(Dataset):
    """Base dataset class."""

    def __init__(self, data: List[Dict], config: DataConfig):
        self.data = data
        self.config = config

        if config.max_samples > 0:
            self.data = self.data[:config.max_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


# ============================================================================
# SQUAD DATASET
# ============================================================================

class SQuADDataset(BaseDataset):
    """SQuAD dataset for question answering."""

    def __init__(self, split: str = "train", config: Optional[DataConfig] = None):
        if config is None:
            config = DataConfig()

        # Load data
        if HAS_DATASETS:
            dataset = load_dataset("squad", split=split)
            data = []
            for item in dataset:
                data.append({
                    "id": item["id"],
                    "context": item["context"],
                    "question": item["question"],
                    "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                    "answer_start": item["answers"]["answer_start"][0] if item["answers"]["answer_start"] else 0,
                })
        else:
            # Fallback: create dummy data
            data = [
                {
                    "id": f"dummy_{i}",
                    "context": f"This is context {i}. It contains information.",
                    "question": f"What is question {i}?",
                    "answer": f"Answer {i}",
                    "answer_start": 0,
                }
                for i in range(100)
            ]

        super().__init__(data, config)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Format as prefix and answer
        prefix = f"Context: {item['context']}\nQuestion: {item['question']}\n"

        # Truncate if needed
        if len(prefix) > self.config.max_prefix_len:
            prefix = prefix[:self.config.max_prefix_len]

        answer = item["answer"]
        if len(answer) > self.config.max_answer_len:
            answer = answer[:self.config.max_answer_len]

        # Get anchor text (with fallback)
        anchor_text = "Answer: "
        if hasattr(self.config, 'compression') and self.config.compression:
            if hasattr(self.config.compression, 'anchor_text'):
                anchor_text = self.config.compression.anchor_text

        return {
            "id": item["id"],
            "prefix": prefix,
            "answer": answer,
            "full_text": prefix + anchor_text + answer,
        }


# ============================================================================
# SST2 DATASET
# ============================================================================

class SST2Dataset(BaseDataset):
    """SST-2 sentiment classification dataset."""

    def __init__(self, split: str = "train", config: Optional[DataConfig] = None):
        if config is None:
            config = DataConfig()

        # Load data
        if HAS_DATASETS:
            dataset = load_dataset("sst2", split=split if split != "test" else "validation")
            data = []
            for item in dataset:
                data.append({
                    "text": item["sentence"],
                    "label": item["label"],
                })
        else:
            # Fallback: create dummy data
            data = [
                {
                    "text": f"This is {'positive' if i % 2 == 0 else 'negative'} text {i}.",
                    "label": i % 2,
                }
                for i in range(100)
            ]

        super().__init__(data, config)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Format for generation
        prefix = f"Classify the sentiment: {item['text']}\n"
        answer = "positive" if item["label"] == 1 else "negative"

        # Get anchor text (with fallback)
        anchor_text = "Answer: "
        if hasattr(self.config, 'compression') and self.config.compression:
            if hasattr(self.config.compression, 'anchor_text'):
                anchor_text = self.config.compression.anchor_text

        return {
            "id": str(idx),
            "prefix": prefix,
            "answer": answer,
            "label": item["label"],
            "full_text": prefix + anchor_text + answer,
        }


# ============================================================================
# AGNEWS DATASET
# ============================================================================

class AGNewsDataset(BaseDataset):
    """AG News classification dataset."""

    LABEL_MAP = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology"
    }

    def __init__(self, split: str = "train", config: Optional[DataConfig] = None):
        if config is None:
            config = DataConfig()

        # Load data
        if HAS_DATASETS:
            dataset = load_dataset("ag_news", split=split if split != "val" else "test")
            data = []
            for item in dataset:
                data.append({
                    "text": item["text"],
                    "label": item["label"],
                })
        else:
            # Fallback: create dummy data
            data = [
                {
                    "text": f"News article {i} about {self.LABEL_MAP[i % 4]}.",
                    "label": i % 4,
                }
                for i in range(100)
            ]

        super().__init__(data, config)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Format for generation
        prefix = f"Classify the news category: {item['text']}\n"
        answer = self.LABEL_MAP[item["label"]]

        # Get anchor text (with fallback)
        anchor_text = "Answer: "
        if hasattr(self.config, 'compression') and self.config.compression:
            if hasattr(self.config.compression, 'anchor_text'):
                anchor_text = self.config.compression.anchor_text

        return {
            "id": str(idx),
            "prefix": prefix,
            "answer": answer,
            "label": item["label"],
            "full_text": prefix + anchor_text + answer,
        }


# ============================================================================
# DATA COLLATOR
# ============================================================================

class DataCollator:
    """Custom data collator for batching."""

    def __init__(self, tokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        # Extract fields
        prefixes = [item["prefix"] for item in batch]
        answers = [item["answer"] for item in batch]
        full_texts = [item["full_text"] for item in batch]

        # Tokenize
        prefix_encodings = self.tokenizer(
            prefixes,
            padding=True,
            truncation=True,
            max_length=self.config.max_prefix_len,
            return_tensors="pt"
        )

        answer_encodings = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.config.max_answer_len,
            return_tensors="pt"
        )

        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prefix_len + self.config.max_answer_len,
            return_tensors="pt"
        )

        return {
            "prefix_input_ids": prefix_encodings["input_ids"],
            "prefix_attention_mask": prefix_encodings["attention_mask"],
            "answer_input_ids": answer_encodings["input_ids"],
            "answer_attention_mask": answer_encodings["attention_mask"],
            "full_input_ids": full_encodings["input_ids"],
            "full_attention_mask": full_encodings["attention_mask"],
            "labels": full_encodings["input_ids"].clone(),
        }


# ============================================================================
# DATASET REGISTRY
# ============================================================================

def get_dataset(name: str, split: str = "train", config: Optional[DataConfig] = None) -> Dataset:
    """Get dataset by name."""
    datasets = {
        "squad": SQuADDataset,
        "sst2": SST2Dataset,
        "agnews": AGNewsDataset,
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")

    return datasets[name](split=split, config=config)


def get_dataloader(
    dataset: Dataset,
    tokenizer,
    config: DataConfig,
    shuffle: bool = True,
    distributed: bool = False
) -> DataLoader:
    """Create dataloader."""
    collator = DataCollator(tokenizer, config)

    # Get batch size (with fallback)
    batch_size = 32
    if hasattr(config, 'training') and config.training:
        if hasattr(config.training, 'batch_size'):
            batch_size = config.training.batch_size

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collator,
        sampler=sampler,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )


# ============================================================================
# EXTENDED DATA UTILITIES (from Section 17)
# ============================================================================

def normalize_space(text: str) -> str:
    """Normalize whitespace in text."""
    import re
    return re.sub(r"\s+", " ", text).strip()


def load_hotpot_subset(
    split: str = "train",
    samples: int = 128,
    seed: int = 0,
    config: str = "fullwiki",
    k_sent: int = 4,
    max_items: int = 3,
    neighbor: int = 1,
    use_supporting: bool = True,
    max_chars: int = 2000,
) -> List[Dict[str, Any]]:
    """Load HotpotQA subset with proper context building."""
    if not HAS_DATASETS:
        return []

    from datasets import load_dataset

    try:
        dataset = load_dataset("hotpot_qa", config, split=split)
    except Exception:
        if config != "distractor":
            dataset = load_dataset("hotpot_qa", "distractor", split=split)
        else:
            raise

    if samples > 0:
        n = len(dataset)
        indices = list(range(n))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(indices)
        indices = indices[:samples]
        dataset = dataset.select(indices)

    res = []
    for ex in dataset:
        context_text = build_context_text(
            ex,
            k_sent=k_sent,
            max_items=max_items,
            neighbor=neighbor,
            use_supporting=use_supporting,
            max_chars=max_chars
        )

        question = normalize_space(ex["question"])
        answer = normalize_space(ex["answer"])

        if context_text and question and answer:
            source = f"Context: {context_text}\n\nQuestion: {question}"
            res.append({"source": source, "answer": answer})

    return res


def build_context_text(
    ex: Dict[str, Any],
    k_sent: int = 4,
    max_items: int = 3,
    neighbor: int = 1,
    use_supporting: bool = True,
    max_chars: int = 2000
) -> str:
    """Build context text from HotpotQA example."""
    title_to_sents = index_context(ex.get("context", []))
    pieces = []

    if use_supporting:
        pieces.extend(gather_supporting_facts(ex, title_to_sents, neighbor=neighbor, max_sf=6))

    if len(pieces) < max_items:
        fallback = fallback_context(title_to_sents, max_items=max_items, k_sent=k_sent)
        pieces.extend(fallback)

    text = normalize_space(" ".join([p for p in pieces if p]))
    return text[:max_chars]


def index_context(ctx) -> Dict[str, List[str]]:
    """Index context into {title -> [sentences]} mapping."""
    title_to_sents = {}

    if isinstance(ctx, dict):
        titles = ctx.get("title") or ctx.get("titles") or []
        sents_lists = ctx.get("sentences") or []
        if isinstance(titles, list) and isinstance(sents_lists, list):
            for t, sents in zip(titles, sents_lists):
                title_to_sents[str(t)] = [str(s) for s in (sents or [])]

    elif isinstance(ctx, list):
        for item in ctx:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], list):
                title, sents = item[0], item[1]
                title_to_sents[str(title)] = [str(s) for s in sents]
            elif isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences", [])
                if isinstance(sents, list):
                    title_to_sents[str(title)] = [str(s) for s in sents]

    return title_to_sents


def gather_supporting_facts(
    ex: Dict[str, Any],
    title_to_sents: Dict[str, List[str]],
    neighbor: int = 1,
    max_sf: int = 6
) -> List[str]:
    """Gather supporting facts from context."""
    chunks = []
    sup = ex.get("supporting_facts", None)

    if not sup:
        return chunks

    def add_span(title: str, idx: int):
        sents = title_to_sents.get(title, [])
        if not sents:
            return False
        i = int(idx)
        start = max(0, i - neighbor)
        end = min(len(sents), i + neighbor + 1)
        if start < end:
            chunks.append(normalize_space(" ".join(sents[start:end])))
            return True
        return False

    used = 0
    if isinstance(sup, dict):
        titles = sup.get("title", [])
        sent_ids = sup.get("sent_id", [])
        for t, i in zip(titles, sent_ids):
            if add_span(str(t), int(i)):
                used += 1
            if used >= max_sf:
                break
    elif isinstance(sup, list):
        for item in sup:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if add_span(str(item[0]), int(item[1])):
                    used += 1
            if used >= max_sf:
                break

    return chunks


def fallback_context(
    title_to_sents: Dict[str, List[str]],
    max_items: int = 3,
    k_sent: int = 4
) -> List[str]:
    """Fallback context if supporting facts are missing."""
    chunks = []
    for _, sents in title_to_sents.items():
        if not sents:
            continue
        chunks.append(normalize_space(" ".join(sents[:k_sent])))
        if len(chunks) >= max_items:
            break
    return chunks


def load_squad_subset(
    split: str = "train",
    samples: int = 128,
    seed: int = 0,
    max_chars: int = 2000,
) -> List[Dict[str, Any]]:
    """Load SQuAD subset."""
    if not HAS_DATASETS:
        return []

    from datasets import load_dataset

    dataset = load_dataset("squad", split="validation" if split == "val" else split)

    if samples > 0:
        n = len(dataset)
        indices = list(range(n))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(indices)
        indices = indices[:samples]
        dataset = dataset.select(indices)

    res = []
    for ex in dataset:
        context = normalize_space(ex["context"])[:max_chars]
        question = normalize_space(ex["question"])
        answers = ex["answers"]["text"]
        answer = answers[0] if answers else ""

        source = f"Context: {context}\n\nQuestion: {question}"
        res.append({"source": source, "answer": answer})

    return res