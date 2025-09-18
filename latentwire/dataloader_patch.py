from typing import Optional

from datasets import load_dataset


def patch_dataloader_defaults() -> None:
    """Placeholder for dataloader tweaks used during evaluation.

    Older scripts imported this helper to apply multiprocessing or shuffle
    patches. The current pipeline does not require any mutations, but we keep
    the function around for backwards compatibility so existing entrypoints do
    not crash when they attempt to call it.
    """
    return None


def load_squad_split(split: str = "train", samples: Optional[int] = None):
    ds = load_dataset("squad", split=split)
    if samples is not None:
        ds = ds.select(range(min(len(ds), samples)))
    return ds
