from datasets import load_dataset

def load_squad_split(split="train", samples: int = None):
    ds = load_dataset("squad", split=split)
    if samples is not None:
        ds = ds.select(range(min(len(ds), samples)))
    return ds
