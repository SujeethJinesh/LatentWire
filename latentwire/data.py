from typing import List, Dict, Any
import random
import re

from datasets import load_dataset, Dataset

def _first_sentences(paragraph: List[str], k: int = 2) -> str:
    sents = paragraph[:k]
    text = " ".join(sents)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _load_hotpot(split: str, config: str) -> Dataset:
    try:
        return load_dataset("hotpot_qa", config, split=split)
    except Exception:
        if config != "distractor":
            return load_dataset("hotpot_qa", "distractor", split=split)
        raise

def load_hotpot_subset(split: str = "train", samples: int = 128, seed: int = 0, config: str = "fullwiki") -> List[Dict[str, Any]]:
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
        ctx_items = ex.get("context", [])
        pieces = []
        try:
            for item in ctx_items[:2]:
                if isinstance(item, dict):
                    sents = item.get("sentences", [])
                else:
                    sents = item[1] if len(item) > 1 else []
                pieces.append(_first_sentences(sents, k=2))
        except Exception:
            pass
        ctx = " ".join([p for p in pieces if p])[:1000]
        source = f"Question: {q}\nContext: {ctx}\nAnswer:"
        examples.append({"source": source, "answer": ans})
    return examples
