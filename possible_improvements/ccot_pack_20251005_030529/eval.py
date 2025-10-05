
# eval.py (drop-in)
# - Ensures correct chat templating: no double special tokens
# - Provides evaluate_batch(...) that uses tokenizer.apply_chat_template(add_generation_prompt=True)
# - Keeps flags minimal; integrates smoothly with train-time features

import torch
from transformers import AutoTokenizer
import re

def clean_pred(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(assistant|assistant:|Assistant:)\s*", "", s)
    s = re.sub(r"^\s*(answer\s*:)\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

@torch.no_grad()
def generate_from_messages(model, tokenizer, messages, max_new_tokens=64, temperature=0.0, top_p=1.0, device="cuda"):
    # serialize with chat template
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def evaluate_batch(model, tokenizer, batch):
    # batch: dict with "question", "context", "answers"
    messages = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":f"Question: {batch['question']}\nContext: {batch['context']}\n"}
    ]
    pred = generate_from_messages(model, tokenizer, messages)
    pred = clean_pred(pred)
    gold = batch.get("answers", "").strip()
    return pred, gold
