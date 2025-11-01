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
    ds = load_dataset(name, split=split)  # Just "squad", not "rajpurkar/squad"
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:samples]
    out = []
    for i in idxs:
        ex = ds[i]
        # answers field is a dict-like object with 'text' field containing list of answers
        try:
            if "answers" in ex and ex["answers"] is not None:
                ans_list = ex["answers"]["text"] if isinstance(ex["answers"], dict) else ex["answers"].get("text", [])
                answer = ans_list[0] if ans_list and len(ans_list) > 0 else ""
            else:
                answer = ""
        except (KeyError, IndexError, TypeError):
            answer = ""

        context = ex.get("context", "") if isinstance(ex, dict) else ex["context"]
        question = ex.get("question", "") if isinstance(ex, dict) else ex["question"]
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

def load_examples(dataset: str = "hotpot", **kwargs) -> List[Dict[str, Any]]:
    """
    Unified front: dataset ∈ {"hotpot","squad","squad_v2","gsm8k"}
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
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose hotpot|squad|squad_v2|gsm8k")
