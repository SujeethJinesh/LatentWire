import os
import json
import argparse

import torch

from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer
from latentwire.data import load_hotpot_subset


def collate_bytes(texts, byte_tok: ByteTokenizer, device: str):
    ids = [byte_tok.encode(t) for t in texts]
    maxT = max([x.size(0) for x in ids])
    batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.size(0), dtype=torch.long)], dim=0) for x in ids], dim=0)
    return batch.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--llama_id", type=str, default=None)
    ap.add_argument("--qwen_id", type=str, default=None)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with open(os.path.join(args.ckpt, "config.json")) as f:
        cfg = json.load(f)
    llama_id = args.llama_id or cfg["llama_id"]
    qwen_id  = args.qwen_id  or cfg["qwen_id"]

    llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))

    encoder = InterlinguaEncoder(d_z=cfg["d_z"], latent_len=cfg["latent_len"])
    encoder.load_state_dict(torch.load(os.path.join(args.ckpt, "encoder.pt"), map_location="cpu"))
    encoder.to(device).eval()

    adp_llama = Adapter(d_z=cfg["d_z"], d_model=llama.d_model)
    adp_llama.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location="cpu"))
    adp_llama.to(device).eval()

    adp_qwen = Adapter(d_z=cfg["d_z"], d_model=qwen.d_model)
    adp_qwen.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_qwen.pt"), map_location="cpu"))
    adp_qwen.to(device).eval()

    examples = load_hotpot_subset(split="validation", samples=args.samples, seed=123)
    texts = [e["source"] for e in examples]
    answers = [e["answer"] for e in examples]

    byte_tok = ByteTokenizer(max_bytes=cfg["byte_max"])
    z_bytes = collate_bytes(texts, byte_tok, device)
    with torch.no_grad():
        z = encoder(z_bytes)
        prefix_llama = adp_llama(z)
        prefix_qwen  = adp_qwen(z)

        out_ids_llama = llama.generate_from_prefix(prefix_llama, max_new_tokens=args.max_new_tokens, temperature=0.0)
        out_ids_qwen  = qwen.generate_from_prefix(prefix_qwen,  max_new_tokens=args.max_new_tokens, temperature=0.0)

    for i in range(len(texts)):
        print("="*80)
        print(f"Example {i+1}")
        print(texts[i][:240].replace("\n", " ") + ("..." if len(texts[i])>240 else ""))
        print("\nGold answer:", answers[i])
        s_llama = llama.tokenizer.decode(out_ids_llama[i], skip_special_tokens=True)
        s_qwen  = qwen.tokenizer.decode(out_ids_qwen[i],  skip_special_tokens=True)
        print("\n[Llama output]")
        print(s_llama)
        print("\n[Qwen output]")
        print(s_qwen)

if __name__ == "__main__":
    main()

