# latentwire/data_pipeline.py
"""
Data loading utilities used by the training loop.

This module consolidates the dataset preparation that previously lived
inline inside `latentwire.train`. Behaviour is identical: we emit the
same informational prints and rely on `latentwire.data.load_examples`
for dataset access.
"""
from typing import List, Tuple

from latentwire.data import load_examples


def prepare_training_data(
    dataset: str,
    samples: int,
    data_seed: int,
    hotpot_config: str = "fullwiki",
) -> Tuple[List[str], List[str]]:
    """
    Load and assemble the (text, answer) pairs for the training loop.

    Args:
        dataset: Name of the dataset ("squad", "squad_v2", or "hotpot").
        samples: Number of samples to draw.
        data_seed: Seed used for sampling.
        hotpot_config: HotpotQA configuration (defaults to "fullwiki").

    Returns:
        texts: List of prompt strings.
        answers: List of answer strings aligned with `texts`.
    """
    print("Loading dataset subset...")

    if dataset.startswith("squad"):
        print("Loading SQuAD subset...")
        examples = load_examples(
            dataset=dataset,
            split="train",
            samples=samples,
            seed=data_seed,
        )
    else:
        print("Loading HotpotQA subset...")
        examples = load_examples(
            dataset="hotpot",
            split="train",
            samples=samples,
            seed=data_seed,
            config=hotpot_config,
        )

    if not examples:
        raise RuntimeError("No training examples loaded.")

    texts = [ex["source"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    return texts, answers

