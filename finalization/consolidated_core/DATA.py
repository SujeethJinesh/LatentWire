#!/usr/bin/env python
"""
Consolidated Data Module for LatentWire

This module combines all data-related functionality including:
- Dataset loading (SQuAD, HotpotQA, GSM8K, TREC, AG News, SST-2, XSUM)
- Metrics computation (EM, F1, NLL, accuracy, ROUGE)
- Evaluation utilities and helpers
- Data preprocessing and normalization

All data loading, metrics, and evaluation functionality is consolidated here
for centralized management and easier maintenance.
"""

import re
import string
import json
import time
import warnings
import logging
import random
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATASET LOADING
# =============================================================================

def _normalize_space(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------------------------------
# HotpotQA Dataset
# -----------------------------------------------------------------------------

def _load_hotpot(split: str, config: str) -> Dataset:
    """Load HotpotQA dataset with fallback to distractor if fullwiki unavailable."""
    try:
        return load_dataset("hotpot_qa", config, split=split)
    except Exception:
        # Fallback to distractor if fullwiki unavailable on the machine
        if config != "distractor":
            return load_dataset("hotpot_qa", "distractor", split=split)
        raise


def _index_context(ctx) -> Dict[str, List[str]]:
    """
    Build {title -> [sentences]} for both Hotpot formats:
    - dict: {"title": [t1, t2, ...], "sentences": [[...], [...], ...]}
    - list: [[title, [sents...]], ...]  OR [{"title": ..., "sentences": [...]}, ...]
    """
    title_to_sents = {}
    # Case A: dict-of-lists
    if isinstance(ctx, dict):
        titles = ctx.get("title") or ctx.get("titles") or []
        sents_lists = ctx.get("sentences") or []
        if isinstance(titles, list) and isinstance(sents_lists, list):
            for t, sents in zip(titles, sents_lists):
                title_to_sents[str(t)] = [str(s) for s in (sents or [])]
        return title_to_sents
    # Case B: list of pairs or dicts
    if isinstance(ctx, list):
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


def _gather_supporting_facts(
    ex: Dict[str, Any],
    title_to_sents: Dict[str, List[str]],
    neighbor: int = 1,
    max_sf: int = 6
) -> List[str]:
    """
    Use supporting_facts when present:
    - dict: {"title": [...], "sent_id": [...]}
    - list: [[title, idx], ...]
    """
    chunks: List[str] = []
    sup = ex.get("supporting_facts", None)
    if not sup:
        return chunks

    def add_span(title: str, idx: int):
        sents = title_to_sents.get(title, [])
        if not sents:
            return False
        i = int(idx)
        start = max(0, i - neighbor)
        end   = min(len(sents), i + neighbor + 1)
        if start < end:
            chunks.append(_normalize_space(" ".join(sents[start:end])))
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


def _fallback_context(
    title_to_sents: Dict[str, List[str]],
    max_items: int = 3,
    k_sent: int = 4
) -> List[str]:
    """Fallback if supporting facts are missing: take the first few titles and first k sentences each."""
    chunks: List[str] = []
    for _, sents in title_to_sents.items():
        if not sents:
            continue
        chunks.append(_normalize_space(" ".join(sents[:k_sent])))
        if len(chunks) >= max_items:
            break
    return chunks


def build_context_text(
    ex: Dict[str, Any],
    k_sent: int = 4,
    max_items: int = 3,
    neighbor: int = 1,
    use_supporting: bool = True,
    max_chars: int = 2000
) -> str:
    """Build context text from HotpotQA example."""
    title_to_sents = _index_context(ex.get("context", []))
    pieces: List[str] = []

    if use_supporting:
        pieces.extend(_gather_supporting_facts(ex, title_to_sents, neighbor=neighbor, max_sf=6))

    # Ensure we have at least some context
    if len(pieces) < max_items:
        # add fallback items not already included
        fallback = _fallback_context(title_to_sents, max_items=max_items, k_sent=k_sent)
        pieces.extend(fallback)

    text = _normalize_space(" ".join([p for p in pieces if p]))
    return text[:max_chars]


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
    """
    Return a list of {source, answer} with a compact but informative context.
    - Uses supporting_facts when available; otherwise falls back to first titles/sentences.
    - Context length is clamped by max_chars to keep things small.
    """
    ds = _load_hotpot(split, config)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:samples]

    examples = []
    for i in idxs:
        ex = ds[i]
        q = ex.get("question", "")
        ans = ex.get("answer", "")
        ctx = build_context_text(
            ex,
            k_sent=k_sent,
            max_items=max_items,
            neighbor=neighbor,
            use_supporting=use_supporting,
            max_chars=max_chars,
        )
        source = f"Question: {q}\nContext: {ctx}\n"
        examples.append({"source": source, "answer": ans})
    return examples


# -----------------------------------------------------------------------------
# SQuAD Dataset (v1 and v2)
# -----------------------------------------------------------------------------

def load_squad_subset(split: str = "train", samples: int = 512, seed: int = 0, v2: bool = False) -> List[Dict[str, Any]]:
    """Load SQuAD v1 or v2 dataset subset."""
    name = "squad_v2" if v2 else "squad"
    # Load dataset - explicitly specify plain_text=False for older datasets library
    try:
        ds = load_dataset(f"rajpurkar/{name}", split=split)
    except TypeError:
        # Fallback for older datasets library that requires trust_remote_code
        ds = load_dataset(f"rajpurkar/{name}", split=split, trust_remote_code=True)

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:samples]
    out = []
    for i in idxs:
        ex = ds[i]
        # HuggingFace datasets return dict-like objects with direct key access
        context = str(ex["context"])
        question = str(ex["question"])
        # answers is a dict with 'text' key containing a list of answer strings
        # Handle both dict and object access patterns
        answers_obj = ex["answers"]
        if hasattr(answers_obj, "get"):
            ans_list = answers_obj.get("text", [])
        elif hasattr(answers_obj, "text"):
            ans_list = answers_obj.text
        else:
            # Fallback: try dictionary access
            ans_list = answers_obj["text"] if "text" in answers_obj else []

        answer = str(ans_list[0]) if ans_list else ""

        source = f"Context: {context}\nQuestion: {question}\n"
        out.append({"source": source, "answer": answer})
    return out


# -----------------------------------------------------------------------------
# GSM8K Math Reasoning Dataset
# -----------------------------------------------------------------------------

def load_gsm8k_subset(split: str = "train", samples: int = 512, seed: int = 0) -> List[Dict[str, Any]]:
    """Load GSM8K math reasoning dataset."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:samples]
    out = []
    for i in idxs:
        ex = ds[i]
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        # GSM8K answers are in format: "explanation #### numerical_answer"
        # We'll keep full answer for now, extraction happens in metrics
        source = f"Question: {question}\n"
        out.append({"source": source, "answer": answer})
    return out


# 8-Shot CoT Examples for GSM8K (Standard from lm-evaluation-harness)
GSM8K_COT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    }
]


def format_gsm8k_prompt(question: str, few_shot: bool = True) -> str:
    """Format a GSM8K question with optional few-shot examples."""
    if few_shot:
        prompt = ""
        for ex in GSM8K_COT_EXAMPLES:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt
    else:
        return f"Question: {question}\nAnswer:"


def extract_numerical_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from GSM8K-style text.

    Handles two formats:
    1. Dataset format: "explanation #### numerical_answer"
    2. CoT format: "The answer is {number}"

    Returns normalized numerical string or None if not found.
    """
    if not text:
        return None

    # Try dataset format first (#### pattern)
    dataset_match = re.search(r'####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if dataset_match:
        answer = dataset_match.group(1)
        # Remove commas from numbers
        return answer.replace(',', '')

    # Try CoT format (The answer is X)
    cot_match = re.search(r'[Tt]he answer is[:\s]+([+-]?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if cot_match:
        answer = cot_match.group(1)
        # Remove commas from numbers
        return answer.replace(',', '')

    # Try to find any standalone number at the end
    end_number = re.search(r'([+-]?\d+(?:,\d+)*(?:\.\d+)?)\s*\.?\s*$', text)
    if end_number:
        answer = end_number.group(1)
        return answer.replace(',', '')

    return None


def compute_gsm8k_accuracy(predictions: List[str], gold_answers: List[str]) -> Dict[str, float]:
    """
    Compute GSM8K accuracy by extracting and comparing numerical answers.

    Returns:
        Dict with accuracy and per-sample results
    """
    correct = 0
    results = []

    for pred, gold in zip(predictions, gold_answers):
        pred_num = extract_numerical_answer(pred)
        gold_num = extract_numerical_answer(gold)

        is_correct = False
        if pred_num and gold_num:
            # Normalize both to float for comparison
            try:
                pred_float = float(pred_num)
                gold_float = float(gold_num)
                # Use small epsilon for float comparison
                is_correct = abs(pred_float - gold_float) < 1e-6
            except ValueError:
                # Fallback to string comparison
                is_correct = pred_num == gold_num

        if is_correct:
            correct += 1

        results.append({
            'prediction': pred,
            'gold': gold,
            'pred_number': pred_num,
            'gold_number': gold_num,
            'correct': is_correct
        })

    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(predictions),
        'per_sample': results
    }


# -----------------------------------------------------------------------------
# TREC Question Classification Dataset
# -----------------------------------------------------------------------------

def load_trec_subset(split: str = "test", samples: int = None, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Load TREC Question Classification dataset.

    Dataset: SetFit/TREC-QC (modern parquet format)
    Task: Question classification into 6 coarse categories

    Splits:
        - train: 5,452 examples
        - test: 500 examples

    Coarse labels (6 classes):
        0: ABBR - Abbreviation
        1: ENTY - Entity
        2: DESC - Description and abstract concepts
        3: HUM  - Human being
        4: LOC  - Location
        5: NUM  - Numeric value

    Returns examples with:
        - source: "Question: {question}\n"
        - answer: label name (e.g., "ABBR", "ENTY", etc.)

    Standard evaluation uses the full test set (500 examples) with accuracy metric.
    """
    # Load from SetFit (modern parquet format, no loading scripts)
    ds = load_dataset("SetFit/TREC-QC", split=split)

    # Define label mapping
    LABEL_NAMES = {
        0: "ABBR",  # Abbreviation
        1: "ENTY",  # Entity
        2: "DESC",  # Description
        3: "HUM",   # Human
        4: "LOC",   # Location
        5: "NUM"    # Numeric
    }

    # For TREC, standard evaluation uses the full test set
    # If samples is None or >= dataset size, use all examples
    if samples is None or samples >= len(ds):
        idxs = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:samples]

    out = []
    for i in idxs:
        ex = ds[i]
        question = ex["text"]
        # Use coarse label for classification
        label_id = ex["label_coarse"]
        label_name = LABEL_NAMES[label_id]

        source = f"Question: {question}\n"
        out.append({"source": source, "answer": label_name})

    return out


# -----------------------------------------------------------------------------
# AG News Topic Classification Dataset
# -----------------------------------------------------------------------------

def load_agnews_subset(split: str = "test", samples: int = None, seed: int = 0, max_chars: int = 256) -> List[Dict[str, Any]]:
    """
    Load AG News topic classification dataset.

    Dataset: ag_news
    Task: Topic classification into 4 categories

    Splits:
        - train: 120,000 examples
        - test: 7,600 examples (1,900 per class)

    Labels (4 classes):
        0: World
        1: Sports
        2: Business
        3: Sci/Tech (Science and Technology)

    Returns examples with:
        - source: "Article: {text[:max_chars]}\nTopic (world, sports, business, or science):"
        - answer: label name (e.g., "world", "sports", etc.)

    Standard evaluation uses the full test set (7,600 examples) with accuracy metric.
    """
    ds = load_dataset("ag_news", split=split)

    # Define label mapping
    LABEL_NAMES = {
        0: "world",
        1: "sports",
        2: "business",
        3: "science"  # AG News uses "Sci/Tech", we normalize to "science"
    }

    # For AG News, standard evaluation uses the full test set
    # If samples is None or >= dataset size, use all examples
    if samples is None or samples >= len(ds):
        idxs = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:samples]

    out = []
    for i in idxs:
        ex = ds[i]
        text = _normalize_space(ex["text"])
        label_id = ex["label"]
        label_name = LABEL_NAMES[label_id]

        # Truncate article to max_chars
        truncated_text = text[:max_chars]
        source = f"Article: {truncated_text}\nTopic (world, sports, business, or science):"
        out.append({"source": source, "answer": label_name})

    return out


# -----------------------------------------------------------------------------
# SST-2 Sentiment Classification Dataset
# -----------------------------------------------------------------------------

def load_sst2_subset(split: str = "validation", samples: int = None, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Load SST-2 (Stanford Sentiment Treebank v2) classification dataset.

    Dataset: SST-2 from GLUE benchmark
    Task: Binary sentiment classification

    Splits:
        - train: 67,349 examples
        - validation: 872 examples
        - test: 1,821 examples (labels not available)

    Labels (2 classes):
        0: negative
        1: positive

    Returns examples with:
        - source: "Classify sentiment as 'positive' or 'negative': {sentence}\nSentiment:"
        - answer: label name (e.g., "positive", "negative")

    Standard evaluation uses the full validation set (872 examples) with accuracy metric.
    """
    # Load from GLUE benchmark
    ds = load_dataset("glue", "sst2", split=split)

    # Define label mapping
    LABEL_NAMES = {
        0: "negative",
        1: "positive"
    }

    # For SST-2, standard evaluation uses the full validation set
    # If samples is None or >= dataset size, use all examples
    if samples is None or samples >= len(ds):
        idxs = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:samples]

    out = []
    for i in idxs:
        ex = ds[i]
        sentence = _normalize_space(ex["sentence"])
        label_id = ex["label"]
        label_name = LABEL_NAMES[label_id]

        source = f"Classify sentiment as 'positive' or 'negative': {sentence}\nSentiment:"
        out.append({"source": source, "answer": label_name})

    return out


# -----------------------------------------------------------------------------
# XSUM Summarization Dataset
# -----------------------------------------------------------------------------

def load_xsum_subset(split: str = "test", samples: int = None, seed: int = 0, max_chars: int = 512) -> List[Dict[str, Any]]:
    """
    Load XSUM abstractive summarization dataset.

    Dataset: XSUM (BBC articles)
    Task: Single-sentence abstractive summarization

    Splits:
        - train: 204,045 examples
        - validation: 11,332 examples
        - test: 11,334 examples

    Returns examples with:
        - source: "Article: {document[:max_chars]}\nSummary:"
        - answer: summary (single sentence)

    Standard evaluation uses ROUGE scores on the test set.

    Note: Due to datasets library compatibility issues with XSUM,
    this function uses CNN/DailyMail as a fallback when XSUM cannot be loaded.
    CNN/DailyMail is a similar abstractive summarization dataset with:
        - train: 287,113 examples
        - validation: 13,368 examples
        - test: 11,490 examples
    """
    # Try different loading methods for XSUM compatibility
    try:
        # Try standard loading first
        ds = load_dataset("xsum", split=split)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            # Fallback to CNN/DailyMail which is more compatible
            # CNN/DailyMail has train/validation/test splits
            # train: 287,113, validation: 13,368, test: 11,490
            print(f"Warning: XSUM not available, using CNN/DailyMail as fallback for summarization task")

            # Map XSUM splits to CNN/DailyMail splits
            cnn_split = split
            if split == "validation":
                cnn_split = "validation"
            elif split == "test":
                cnn_split = "test"
            else:
                cnn_split = "train"

            try:
                # Try loading CNN/DailyMail from alternative sources
                # First try abisee version
                ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=cnn_split)
            except Exception:
                try:
                    # Try alternative CNN/DailyMail sources
                    ds = load_dataset("cnn_dailymail", "3.0.0", split=cnn_split)
                except Exception:
                    # Final fallback - use a simple mock dataset for testing
                    print(f"Warning: Neither XSUM nor CNN/DailyMail available, using mock data for testing")

                    # Create mock data matching XSUM format
                    mock_examples = []
                    for i in range(100):  # Create 100 mock examples
                        mock_examples.append({
                            "document": f"This is a test article number {i}. It contains some text that would normally be a news article. The content here is just placeholder text for testing purposes.",
                            "summary": f"Test summary for article {i}."
                        })

                    # Convert to dataset-like structure
                    class MockDataset:
                        def __init__(self, examples):
                            self.examples = examples

                        def __len__(self):
                            return len(self.examples)

                        def __getitem__(self, idx):
                            return self.examples[idx]

                    ds = MockDataset(mock_examples)
        else:
            raise

    # If samples is None or >= dataset size, use all examples
    if samples is None or samples >= len(ds):
        idxs = list(range(len(ds)))
    else:
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:samples]

    out = []
    for i in idxs:
        ex = ds[i]

        # Handle different dataset formats (XSUM vs CNN/DailyMail)
        if "document" in ex:
            # XSUM format
            document = _normalize_space(ex["document"])
            summary = _normalize_space(ex["summary"])
        elif "article" in ex:
            # CNN/DailyMail format
            document = _normalize_space(ex["article"])
            # CNN/DailyMail has multi-sentence summaries in "highlights"
            # Take first sentence for consistency with XSUM single-sentence format
            highlights = ex.get("highlights", "")
            summary = _normalize_space(highlights.split("\n")[0] if highlights else "")
        else:
            # Mock dataset or unknown format
            document = _normalize_space(ex.get("document", ex.get("text", "")))
            summary = _normalize_space(ex.get("summary", ex.get("highlights", "")))

        # Truncate document to max_chars
        truncated_doc = document[:max_chars]
        source = f"Article: {truncated_doc}\nSummary:"
        out.append({"source": source, "answer": summary})

    return out


# -----------------------------------------------------------------------------
# Unified Dataset Loading Interface
# -----------------------------------------------------------------------------

def load_examples(dataset: str = "hotpot", **kwargs) -> List[Dict[str, Any]]:
    """
    Unified front: dataset ∈ {"hotpot","squad","squad_v2","gsm8k","trec","agnews","sst2","xsum"}
    kwargs forwarded to the underlying loader (split, samples, seed, config, ...)
    """
    ds = dataset.lower()
    if ds == "hotpot":
        return load_hotpot_subset(**kwargs)
    elif ds == "squad":
        return load_squad_subset(v2=False, **{k:v for k,v in kwargs.items() if k!="config"})
    elif ds in ("squad_v2", "squad2"):
        return load_squad_subset(v2=True, **{k:v for k,v in kwargs.items() if k!="config"})
    elif ds == "gsm8k":
        return load_gsm8k_subset(**{k:v for k,v in kwargs.items() if k!="config"})
    elif ds == "trec":
        return load_trec_subset(**{k:v for k,v in kwargs.items() if k!="config"})
    elif ds in ("agnews", "ag_news"):
        return load_agnews_subset(**{k:v for k,v in kwargs.items() if k!="config"})
    elif ds in ("sst2", "sst-2"):
        return load_sst2_subset(**{k:v for k,v in kwargs.items() if k!="config"})
    elif ds == "xsum":
        return load_xsum_subset(**{k:v for k,v in kwargs.items() if k!="config"})
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose hotpot|squad|squad_v2|gsm8k|trec|agnews|sst2|xsum")


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

# -----------------------------------------------------------------------------
# Text Normalization and Basic Metrics
# -----------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(pred: str, gold: str) -> float:
    """Compute F1 score between prediction and gold answer."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()

    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_em(pred: str, gold: str) -> float:
    """Compute Exact Match score."""
    return float(normalize_answer(pred) == normalize_answer(gold))


# -----------------------------------------------------------------------------
# Neural Network Metrics
# -----------------------------------------------------------------------------

def compute_nll_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Tuple[float, int]:
    """Compute negative log-likelihood per token.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Label value to ignore in loss computation

    Returns:
        Tuple of (total_nll, num_tokens)
    """
    # Flatten for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross entropy
    losses = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction='none'
    )

    # Count valid tokens
    valid_mask = (labels_flat != ignore_index)
    num_tokens = valid_mask.sum().item()

    if num_tokens == 0:
        return 0.0, 0

    # Sum losses for valid tokens
    total_nll = losses[valid_mask].sum().item()

    return total_nll, num_tokens


def compute_token_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Compute token-level accuracy.

    Args:
        predictions: Predicted token IDs [batch_size, seq_len]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Label value to ignore

    Returns:
        Token accuracy as a float
    """
    valid_mask = (labels != ignore_index)

    if valid_mask.sum() == 0:
        return 0.0

    correct = (predictions == labels) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().float()

    return accuracy.item()


def compute_first_token_accuracy(
    predictions: List[str],
    gold_answers: List[str],
    tokenizer=None
) -> float:
    """Compute first-token accuracy for generation tasks.

    Args:
        predictions: List of predicted strings
        gold_answers: List of gold answer strings
        tokenizer: Optional tokenizer for token-level comparison

    Returns:
        First-token accuracy as a float
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        if tokenizer is not None:
            # Token-level comparison
            pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
            gold_tokens = tokenizer.encode(gold, add_special_tokens=False)

            if len(pred_tokens) > 0 and len(gold_tokens) > 0:
                if pred_tokens[0] == gold_tokens[0]:
                    correct += 1
        else:
            # Character-level comparison (fallback)
            pred_words = pred.strip().split()
            gold_words = gold.strip().split()

            if len(pred_words) > 0 and len(gold_words) > 0:
                if pred_words[0].lower() == gold_words[0].lower():
                    correct += 1

    return correct / len(predictions)


# -----------------------------------------------------------------------------
# ROUGE Metrics for Summarization
# -----------------------------------------------------------------------------

# Try multiple ROUGE implementations for robustness
ROUGE_BACKEND = None

try:
    from rouge_score import rouge_scorer
    from rouge_score.scoring import Score
    ROUGE_BACKEND = 'rouge_score'
    logger.info("Using rouge_score library for ROUGE computation")
except ImportError:
    logger.warning("rouge_score not available, trying evaluate library...")
    try:
        import evaluate
        ROUGE_BACKEND = 'evaluate'
        logger.info("Using evaluate library for ROUGE computation")
    except ImportError:
        logger.warning("No ROUGE library available. ROUGE metrics will not be available.")
        ROUGE_BACKEND = None

# Suppress tokenization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rouge_score")


@dataclass
class XSumRougeResults:
    """Container for XSUM-specific ROUGE evaluation results."""

    # Core ROUGE metrics
    rouge1_f1: float
    rouge1_precision: float
    rouge1_recall: float

    rouge2_f1: float
    rouge2_precision: float
    rouge2_recall: float

    rougeL_f1: float
    rougeL_precision: float
    rougeL_recall: float

    rougeLsum_f1: float
    rougeLsum_precision: float
    rougeLsum_recall: float

    # Statistical measures
    rouge1_f1_std: float = 0.0
    rouge2_f1_std: float = 0.0
    rougeL_f1_std: float = 0.0
    rougeLsum_f1_std: float = 0.0

    # Confidence intervals
    rouge1_f1_ci: Optional[Tuple[float, float]] = None
    rouge2_f1_ci: Optional[Tuple[float, float]] = None
    rougeL_f1_ci: Optional[Tuple[float, float]] = None
    rougeLsum_f1_ci: Optional[Tuple[float, float]] = None

    # Additional metadata
    num_samples: int = 0
    per_sample_scores: Optional[List[Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'metrics': {
                'rouge1': {
                    'f1': self.rouge1_f1,
                    'precision': self.rouge1_precision,
                    'recall': self.rouge1_recall,
                    'f1_std': self.rouge1_f1_std,
                    'f1_ci': self.rouge1_f1_ci
                },
                'rouge2': {
                    'f1': self.rouge2_f1,
                    'precision': self.rouge2_precision,
                    'recall': self.rouge2_recall,
                    'f1_std': self.rouge2_f1_std,
                    'f1_ci': self.rouge2_f1_ci
                },
                'rougeL': {
                    'f1': self.rougeL_f1,
                    'precision': self.rougeL_precision,
                    'recall': self.rougeL_recall,
                    'f1_std': self.rougeL_f1_std,
                    'f1_ci': self.rougeL_f1_ci
                },
                'rougeLsum': {
                    'f1': self.rougeLsum_f1,
                    'precision': self.rougeLsum_precision,
                    'recall': self.rougeLsum_recall,
                    'f1_std': self.rougeLsum_f1_std,
                    'f1_ci': self.rougeLsum_f1_ci
                }
            },
            'num_samples': self.num_samples,
            'per_sample_scores': self.per_sample_scores
        }

    def print_summary(self):
        """Print a nicely formatted summary of results."""
        print("\n" + "=" * 60)
        print("XSUM ROUGE Evaluation Results")
        print("=" * 60)
        print(f"Number of samples: {self.num_samples}")
        print("\nCore Metrics:")
        print(f"  ROUGE-1 F1: {self.rouge1_f1:.4f} (±{self.rouge1_f1_std:.4f})")
        print(f"  ROUGE-2 F1: {self.rouge2_f1:.4f} (±{self.rouge2_f1_std:.4f})")
        print(f"  ROUGE-L F1: {self.rougeL_f1:.4f} (±{self.rougeL_f1_std:.4f})")
        print(f"  ROUGE-Lsum F1: {self.rougeLsum_f1:.4f} (±{self.rougeLsum_f1_std:.4f})")

        if self.rouge1_f1_ci:
            print("\nConfidence Intervals (95%):")
            print(f"  ROUGE-1 F1: [{self.rouge1_f1_ci[0]:.4f}, {self.rouge1_f1_ci[1]:.4f}]")
            if self.rouge2_f1_ci:
                print(f"  ROUGE-2 F1: [{self.rouge2_f1_ci[0]:.4f}, {self.rouge2_f1_ci[1]:.4f}]")
            if self.rougeL_f1_ci:
                print(f"  ROUGE-L F1: [{self.rougeL_f1_ci[0]:.4f}, {self.rougeL_f1_ci[1]:.4f}]")
            if self.rougeLsum_f1_ci:
                print(f"  ROUGE-Lsum F1: [{self.rougeLsum_f1_ci[0]:.4f}, {self.rougeLsum_f1_ci[1]:.4f}]")
        print("=" * 60)


class XSumRougeScorer:
    """ROUGE scorer optimized for XSUM evaluation."""

    def __init__(self, use_stemmer: bool = True):
        """
        Initialize ROUGE scorer.

        Args:
            use_stemmer: Whether to use Porter stemmer (standard for XSUM)
        """
        self.use_stemmer = use_stemmer
        self.scorer = None

        if ROUGE_BACKEND == 'rouge_score':
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=use_stemmer
            )
        elif ROUGE_BACKEND == 'evaluate':
            self.scorer = evaluate.load('rouge')
        else:
            raise RuntimeError("No ROUGE backend available")

    def score_batch(
        self,
        predictions: List[str],
        references: List[str],
        compute_confidence: bool = False,
        n_bootstrap: int = 1000
    ) -> XSumRougeResults:
        """
        Compute ROUGE scores for a batch of predictions.

        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            compute_confidence: Whether to compute confidence intervals
            n_bootstrap: Number of bootstrap samples for CI

        Returns:
            XSumRougeResults object with all metrics
        """
        if len(predictions) != len(references):
            raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")

        # Compute per-sample scores
        per_sample_scores = []

        if ROUGE_BACKEND == 'rouge_score':
            for pred, ref in zip(predictions, references):
                scores = self.scorer.score(ref, pred)
                per_sample_scores.append({
                    'rouge1_f1': scores['rouge1'].fmeasure,
                    'rouge1_precision': scores['rouge1'].precision,
                    'rouge1_recall': scores['rouge1'].recall,
                    'rouge2_f1': scores['rouge2'].fmeasure,
                    'rouge2_precision': scores['rouge2'].precision,
                    'rouge2_recall': scores['rouge2'].recall,
                    'rougeL_f1': scores['rougeL'].fmeasure,
                    'rougeL_precision': scores['rougeL'].precision,
                    'rougeL_recall': scores['rougeL'].recall,
                    'rougeLsum_f1': scores['rougeLsum'].fmeasure,
                    'rougeLsum_precision': scores['rougeLsum'].precision,
                    'rougeLsum_recall': scores['rougeLsum'].recall
                })
        elif ROUGE_BACKEND == 'evaluate':
            results = self.scorer.compute(
                predictions=predictions,
                references=references,
                use_stemmer=self.use_stemmer
            )
            # evaluate library returns aggregated scores
            # We need to compute per-sample scores separately
            for pred, ref in zip(predictions, references):
                sample_results = self.scorer.compute(
                    predictions=[pred],
                    references=[ref],
                    use_stemmer=self.use_stemmer
                )
                per_sample_scores.append({
                    'rouge1_f1': sample_results['rouge1'],
                    'rouge2_f1': sample_results['rouge2'],
                    'rougeL_f1': sample_results['rougeL'],
                    'rougeLsum_f1': sample_results.get('rougeLsum', sample_results['rougeL'])
                })

        # Aggregate scores
        rouge1_f1_scores = [s['rouge1_f1'] for s in per_sample_scores]
        rouge2_f1_scores = [s['rouge2_f1'] for s in per_sample_scores]
        rougeL_f1_scores = [s['rougeL_f1'] for s in per_sample_scores]
        rougeLsum_f1_scores = [s.get('rougeLsum_f1', s['rougeL_f1']) for s in per_sample_scores]

        result = XSumRougeResults(
            rouge1_f1=np.mean(rouge1_f1_scores),
            rouge1_precision=np.mean([s.get('rouge1_precision', 0) for s in per_sample_scores]),
            rouge1_recall=np.mean([s.get('rouge1_recall', 0) for s in per_sample_scores]),
            rouge2_f1=np.mean(rouge2_f1_scores),
            rouge2_precision=np.mean([s.get('rouge2_precision', 0) for s in per_sample_scores]),
            rouge2_recall=np.mean([s.get('rouge2_recall', 0) for s in per_sample_scores]),
            rougeL_f1=np.mean(rougeL_f1_scores),
            rougeL_precision=np.mean([s.get('rougeL_precision', 0) for s in per_sample_scores]),
            rougeL_recall=np.mean([s.get('rougeL_recall', 0) for s in per_sample_scores]),
            rougeLsum_f1=np.mean(rougeLsum_f1_scores),
            rougeLsum_precision=np.mean([s.get('rougeLsum_precision', 0) for s in per_sample_scores]),
            rougeLsum_recall=np.mean([s.get('rougeLsum_recall', 0) for s in per_sample_scores]),
            rouge1_f1_std=np.std(rouge1_f1_scores),
            rouge2_f1_std=np.std(rouge2_f1_scores),
            rougeL_f1_std=np.std(rougeL_f1_scores),
            rougeLsum_f1_std=np.std(rougeLsum_f1_scores),
            num_samples=len(predictions),
            per_sample_scores=per_sample_scores
        )

        # Compute confidence intervals if requested
        if compute_confidence:
            result.rouge1_f1_ci = self._bootstrap_ci(rouge1_f1_scores, n_bootstrap)
            result.rouge2_f1_ci = self._bootstrap_ci(rouge2_f1_scores, n_bootstrap)
            result.rougeL_f1_ci = self._bootstrap_ci(rougeL_f1_scores, n_bootstrap)
            result.rougeLsum_f1_ci = self._bootstrap_ci(rougeLsum_f1_scores, n_bootstrap)

        return result

    def _bootstrap_ci(
        self,
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            scores: List of scores to bootstrap
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_means = []
        n = len(scores)

        for _ in range(n_bootstrap):
            # Sample with replacement
            sample_indices = np.random.choice(n, n, replace=True)
            sample_scores = [scores[i] for i in sample_indices]
            bootstrap_means.append(np.mean(sample_scores))

        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return (ci_lower, ci_upper)


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True,
    compute_confidence: bool = False
) -> Dict[str, float]:
    """
    Convenience function to compute ROUGE scores.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        use_stemmer: Whether to use Porter stemmer
        compute_confidence: Whether to compute confidence intervals

    Returns:
        Dictionary with ROUGE scores
    """
    if ROUGE_BACKEND is None:
        logger.warning("No ROUGE backend available, returning empty scores")
        return {
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0,
            'rougeLsum_f1': 0.0
        }

    scorer = XSumRougeScorer(use_stemmer=use_stemmer)
    results = scorer.score_batch(predictions, references, compute_confidence=compute_confidence)

    return {
        'rouge1_f1': results.rouge1_f1,
        'rouge2_f1': results.rouge2_f1,
        'rougeL_f1': results.rougeL_f1,
        'rougeLsum_f1': results.rougeLsum_f1,
        'rouge1_f1_std': results.rouge1_f1_std,
        'rouge2_f1_std': results.rouge2_f1_std,
        'rougeL_f1_std': results.rougeL_f1_std,
        'rougeLsum_f1_std': results.rougeLsum_f1_std
    }


# -----------------------------------------------------------------------------
# Metrics Aggregation
# -----------------------------------------------------------------------------

def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple evaluations.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Aggregated metrics with mean and std
    """
    if not metrics_list:
        return {}

    aggregated = {}

    # Collect all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    values.append(value)

        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

    return aggregated


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_qa_predictions(
    predictions: List[str],
    gold_answers: List[str]
) -> Dict[str, float]:
    """
    Evaluate QA predictions with EM and F1 scores.

    Args:
        predictions: List of predicted answers
        gold_answers: List of gold answers

    Returns:
        Dictionary with EM and F1 scores
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")

    em_scores = []
    f1_scores = []

    for pred, gold in zip(predictions, gold_answers):
        em_scores.append(compute_em(pred, gold))
        f1_scores.append(compute_f1(pred, gold))

    return {
        'em': np.mean(em_scores),
        'f1': np.mean(f1_scores),
        'em_std': np.std(em_scores),
        'f1_std': np.std(f1_scores)
    }


def evaluate_classification_predictions(
    predictions: List[str],
    gold_labels: List[str],
    label_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Evaluate classification predictions with accuracy and per-class metrics.

    Args:
        predictions: List of predicted labels
        gold_labels: List of gold labels
        label_names: Optional mapping from label IDs to names

    Returns:
        Dictionary with accuracy and per-class breakdown
    """
    if len(predictions) != len(gold_labels):
        raise ValueError("Predictions and gold labels must have same length")

    correct = 0
    per_class_correct = {}
    per_class_total = {}
    confusion = {}

    for pred, gold in zip(predictions, gold_labels):
        # Normalize predictions and gold labels
        pred_normalized = pred.strip().lower()
        gold_normalized = gold.strip().lower()

        is_correct = pred_normalized == gold_normalized
        if is_correct:
            correct += 1

        # Track per-class metrics
        if gold_normalized not in per_class_total:
            per_class_total[gold_normalized] = 0
            per_class_correct[gold_normalized] = 0
        per_class_total[gold_normalized] += 1

        if is_correct:
            per_class_correct[gold_normalized] += 1

        # Track confusion matrix
        confusion_key = f"{gold_normalized}->{pred_normalized}"
        confusion[confusion_key] = confusion.get(confusion_key, 0) + 1

    # Compute per-class accuracy
    per_class_accuracy = {}
    for label in per_class_total:
        per_class_accuracy[label] = per_class_correct[label] / per_class_total[label]

    return {
        'accuracy': correct / len(predictions) if predictions else 0.0,
        'correct': correct,
        'total': len(predictions),
        'per_class_accuracy': per_class_accuracy,
        'per_class_support': per_class_total,
        'confusion': confusion
    }


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    predictions: Optional[List[Dict[str, Any]]] = None
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary of evaluation metrics
        output_path: Path to save results
        predictions: Optional list of per-sample predictions
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metrics': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    if predictions:
        output['predictions'] = predictions

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")


def print_evaluation_summary(results: Dict[str, Any], dataset_name: str = "Dataset"):
    """
    Print a formatted summary of evaluation results.

    Args:
        results: Dictionary of evaluation metrics
        dataset_name: Name of the dataset for display
    """
    print("\n" + "=" * 60)
    print(f"{dataset_name} Evaluation Results")
    print("=" * 60)

    # Format and print metrics
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"\n  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"    {sub_key}: {sub_value:.4f}")
                else:
                    print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60 + "\n")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Dataset loading
    'load_examples',
    'load_hotpot_subset',
    'load_squad_subset',
    'load_gsm8k_subset',
    'load_trec_subset',
    'load_agnews_subset',
    'load_sst2_subset',
    'load_xsum_subset',

    # GSM8K specific
    'GSM8K_COT_EXAMPLES',
    'format_gsm8k_prompt',
    'extract_numerical_answer',
    'compute_gsm8k_accuracy',

    # Basic metrics
    'normalize_answer',
    'compute_f1',
    'compute_em',

    # Neural network metrics
    'compute_nll_per_token',
    'compute_token_accuracy',
    'compute_first_token_accuracy',

    # ROUGE metrics
    'XSumRougeResults',
    'XSumRougeScorer',
    'compute_rouge_scores',

    # Aggregation
    'aggregate_metrics',

    # Evaluation utilities
    'evaluate_qa_predictions',
    'evaluate_classification_predictions',
    'save_evaluation_results',
    'print_evaluation_summary',
]