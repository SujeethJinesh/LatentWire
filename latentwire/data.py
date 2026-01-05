# latentwire/data.py
from typing import List, Dict, Any, Tuple
import random
import re

from datasets import load_dataset, Dataset

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _load_hotpot(split: str, config: str) -> Dataset:
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
    """
    Fallback if supporting facts are missing: take the first few titles and first k sentences each.
    """
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

# ---- SQuAD loaders (v1 and v2) ----

def load_squad_subset(split: str = "train", samples: int = 512, seed: int = 0, v2: bool = False) -> List[Dict[str, Any]]:
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

    Note: XSUM may require manual download or special handling with newer versions of datasets library.
    If loading fails, consider using:
    - Older version of datasets library (< 2.15)
    - Manual download from https://huggingface.co/datasets/xsum
    """
    # Try different loading methods for XSUM compatibility
    try:
        # Try standard loading first
        ds = load_dataset("xsum", split=split)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            # Fallback for newer datasets library versions
            # This is a known issue with XSUM dataset format
            raise RuntimeError(
                "XSUM dataset requires special handling. "
                "Please either: 1) Use datasets library < 2.15, or "
                "2) Load from a parquet version if available, or "
                "3) Download and convert the dataset manually. "
                "See: https://github.com/huggingface/datasets/issues/5892"
            )
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
        document = _normalize_space(ex["document"])
        summary = _normalize_space(ex["summary"])

        # Truncate document to max_chars
        truncated_doc = document[:max_chars]
        source = f"Article: {truncated_doc}\nSummary:"
        out.append({"source": source, "answer": summary})

    return out

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
