import os
import time
import json
import argparse
from typing import List

import torch

from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
from latentwire.data import load_hotpot_subset
from latentwire.metrics import batch_metrics, _normalize, em, f1


def collate_bytes(texts: List[str], byte_tok: ByteTokenizer, device: str):
    ids = [byte_tok.encode(t) for t in texts]
    maxT = max([x.size(0) for x in ids])
    batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.size(0), dtype=torch.long)], dim=0) for x in ids], dim=0)
    return batch.to(device)


def generate_text_baseline(wrapper: LMWrapper, prompts: List[str], max_new_tokens: int) -> List[str]:
    # Use a manual, framework-agnostic generator to avoid device-specific issues
    out_ids = wrapper.generate_from_text_manual(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    return [wrapper.tokenizer.decode(ids, skip_special_tokens=True) for ids in out_ids]


def generate_latent(wrapper: LMWrapper, prefix_embeds: torch.Tensor, max_new_tokens: int) -> List[str]:
    out_ids = wrapper.generate_from_prefix(prefix_embeds, max_new_tokens=max_new_tokens, temperature=0.0)
    return [wrapper.tokenizer.decode(ids, skip_special_tokens=True) for ids in out_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--llama_id", type=str, default=None)
    ap.add_argument("--qwen_id", type=str, default=None)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--hotpot_config", type=str, default=None, help="Override HotpotQA config (fullwiki/distractor)")
    ap.add_argument("--out_dir", type=str, default=None, help="Write metrics.json/csv here if set")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with open(os.path.join(args.ckpt, "config.json")) as f:
        cfg = json.load(f)
    llama_id = args.llama_id or cfg["llama_id"]
    qwen_id  = args.qwen_id  or cfg["qwen_id"]
    latent_len = cfg["latent_len"]
    encoder_type = cfg.get("encoder_type", "byte")

    llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))

    if encoder_type == "byte":
        encoder = InterlinguaEncoder(d_z=cfg["d_z"], latent_len=cfg["latent_len"]).to(device).eval()
    else:
        encoder = SimpleEncoder(d_z=cfg["d_z"], latent_len=cfg["latent_len"]).to(device).eval()
    encoder.load_state_dict(torch.load(os.path.join(args.ckpt, "encoder.pt"), map_location=device))
    adp_llama = Adapter(d_z=cfg["d_z"], d_model=llama.d_model).to(device).eval()
    adp_llama.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device))
    adp_qwen  = Adapter(d_z=cfg["d_z"], d_model=qwen.d_model).to(device).eval()
    adp_qwen.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_qwen.pt"), map_location=device))

    eval_examples = load_hotpot_subset(split="validation", samples=args.samples, seed=42, config=(args.hotpot_config or "fullwiki"))
    prompts = [e["source"] for e in eval_examples]
    golds   = [e["answer"] for e in eval_examples]

    llama_prompt_tok = llama.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    qwen_prompt_tok  = qwen.tokenizer(prompts,  return_tensors="pt", padding=True, truncation=True)
    avg_prompt_tokens_llama = float((llama_prompt_tok["input_ids"] != llama.tokenizer.pad_token_id).sum().item()) / len(prompts)
    avg_prompt_tokens_qwen  = float((qwen_prompt_tok["input_ids"]  != qwen.tokenizer.pad_token_id).sum().item())  / len(prompts)

    # === Baseline A: Plain text prompting ===
    t0 = time.time()
    llama_text_preds = generate_text_baseline(llama, prompts, max_new_tokens=args.max_new_tokens)
    qwen_text_preds  = generate_text_baseline(qwen,  prompts, max_new_tokens=args.max_new_tokens)
    t_text = time.time() - t0
    llama_text_em, llama_text_f1 = batch_metrics(llama_text_preds, golds)
    qwen_text_em,  qwen_text_f1  = batch_metrics(qwen_text_preds, golds)

    # === Latent prompting (shared interlingua) ===
    with torch.no_grad():
        if encoder_type == "byte":
            byte_tok = ByteTokenizer(max_bytes=cfg["byte_max"])
            z_bytes = collate_bytes(prompts, byte_tok, device)
            Z = encoder(z_bytes)
        else:
            Z = encoder(prompts)
        prefix_llama = adp_llama(Z)
        prefix_qwen  = adp_qwen(Z)

    # Rough payload estimate
    bytes_per_latent = Z.element_size() * Z.size(1) * Z.size(2)

    t0 = time.time()
    llama_latent_preds = generate_latent(llama, prefix_llama, max_new_tokens=args.max_new_tokens)
    qwen_latent_preds  = generate_latent(qwen,  prefix_qwen,  max_new_tokens=args.max_new_tokens)
    t_latent = time.time() - t0

    llama_latent_em, llama_latent_f1 = batch_metrics(llama_latent_preds, golds)
    qwen_latent_em,  qwen_latent_f1  = batch_metrics(qwen_latent_preds, golds)

    # === Token-budget baseline (truncate textual prompt to M tokens) ===
    def truncate_to_k_tokens(tokenizer, texts, k):
        enc = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True, return_attention_mask=False)
        outs = []
        for ids in enc["input_ids"]:
            ids_k = ids[:k]
            outs.append(tokenizer.decode(ids_k, skip_special_tokens=True))
        return outs

    llama_trunc_prompts = truncate_to_k_tokens(llama.tokenizer, prompts, latent_len)
    qwen_trunc_prompts  = truncate_to_k_tokens(qwen.tokenizer,  prompts, latent_len)

    t0 = time.time()
    llama_trunc_preds = generate_text_baseline(llama, llama_trunc_prompts, max_new_tokens=args.max_new_tokens)
    qwen_trunc_preds  = generate_text_baseline(qwen,  qwen_trunc_prompts,  max_new_tokens=args.max_new_tokens)
    t_trunc = time.time() - t0

    llama_trunc_em, llama_trunc_f1 = batch_metrics(llama_trunc_preds, golds)
    qwen_trunc_em,  qwen_trunc_f1  = batch_metrics(qwen_trunc_preds,  golds)

    # === Per-token NLL on gold answers
    def avg_nll_latent(wrapper, prefix, answers, tokenizer):
        tot_nll = 0.0
        tot_tok = 0
        for i, a in enumerate(answers):
            a_ids = tokenizer(a, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            loss = wrapper.forward_with_prefix_loss(prefix[i:i+1], a_ids)
            n_tok = a_ids.size(1) - 1
            tot_nll += float(loss.item()) * n_tok
            tot_tok += int(n_tok)
        return tot_nll / max(1, tot_tok)

    def avg_nll_text(wrapper, prompts_text, answers, tokenizer):
        tot_nll = 0.0
        tot_tok = 0
        for i in range(len(prompts_text)):
            enc_p = tokenizer(prompts_text[i], return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            enc_a = tokenizer(answers[i],      return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            loss, n_tok = wrapper.loss_with_text_prompt(enc_p, enc_a)
            tot_nll += float(loss.item()) * n_tok
            tot_tok += n_tok
        return tot_nll / max(1, tot_tok)

    llama_latent_nll = avg_nll_latent(llama, prefix_llama, golds, llama.tokenizer)
    qwen_latent_nll  = avg_nll_latent(qwen,  prefix_qwen,  golds, qwen.tokenizer)
    llama_text_nll   = avg_nll_text(llama, prompts, golds, llama.tokenizer)
    qwen_text_nll    = avg_nll_text(qwen,  prompts, golds, qwen.tokenizer)

    # === 2-LLM joint (rescored pick on latent runs)
    joint_preds = []
    agree = 0
    for i in range(len(prompts)):
        candA_text = llama_latent_preds[i]
        candB_text = qwen_latent_preds[i]

        A_ids_L = llama.tokenizer(candA_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        A_ids_Q = qwen.tokenizer(candA_text,  return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        B_ids_L = llama.tokenizer(candB_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        B_ids_Q = qwen.tokenizer(candB_text,  return_tensors="pt", add_special_tokens=True).input_ids.to(device)

        scoreA = llama.score_prefix_logprob(prefix_llama[i:i+1], A_ids_L) + qwen.score_prefix_logprob(prefix_qwen[i:i+1], A_ids_Q)
        scoreB = llama.score_prefix_logprob(prefix_llama[i:i+1], B_ids_L) + qwen.score_prefix_logprob(prefix_qwen[i:i+1], B_ids_Q)

        pick = candA_text if scoreA >= scoreB else candB_text
        joint_preds.append(pick)
        if _normalize(candA_text) == _normalize(candB_text):
            agree += 1

    joint_em, joint_f1 = batch_metrics(joint_preds, golds)
    agreement_rate = agree / len(prompts)

    # Oracle bound
    oracle_em = 0.0
    oracle_f1 = 0.0
    for pA, pB, g in zip(llama_latent_preds, qwen_latent_preds, golds):
        oracle_em += max(em(pA, g), em(pB, g))
        oracle_f1 += max(f1(pA, g), f1(pB, g))
    oracle_em /= len(golds)
    oracle_f1 /= len(golds)

    # === Report
    print("\n==== LatentWire Evaluation ====")
    print(f"Samples: {len(prompts)}  |  Max new tokens: {args.max_new_tokens}")
    print(f"Avg prompt tokens (Llama): {avg_prompt_tokens_llama:.1f} | (Qwen): {avg_prompt_tokens_qwen:.1f} | Latent length M: {latent_len}")
    print(f"Compression ratio (Llama): {avg_prompt_tokens_llama/latent_len:.1f}x | (Qwen): {avg_prompt_tokens_qwen/latent_len:.1f}x")
    print(f"Approx interlingua payload per example: {bytes_per_latent} bytes (dtype {str(Z.dtype).split('.')[-1]}, shape M={latent_len}, d_z={cfg['d_z']})\n")

    print("— Baseline: Text prompting")
    print(f"Llama  EM: {llama_text_em:.3f}  F1: {llama_text_f1:.3f}  |  NLL/token (gold): {llama_text_nll:.3f}")
    print(f"Qwen   EM: {qwen_text_em:.3f}   F1: {qwen_text_f1:.3f}   |  NLL/token (gold): {qwen_text_nll:.3f}")
    print(f"Wall clock: {t_text:.2f}s for {len(prompts)} examples")

    print("\n— Latent prompting (shared interlingua)")
    print(f"Llama  EM: {llama_latent_em:.3f}  F1: {llama_latent_f1:.3f}  |  NLL/token (gold): {llama_latent_nll:.3f}")
    print(f"Qwen   EM: {qwen_latent_em:.3f}   F1: {qwen_latent_f1:.3f}   |  NLL/token (gold): {qwen_latent_nll:.3f}")
    print(f"Wall clock: {t_latent:.2f}s for {len(prompts)} examples")

    print("\n— Token-budget baseline (same #prefix tokens as latent)")
    print(f"Llama  EM: {llama_trunc_em:.3f}  F1: {llama_trunc_f1:.3f}")
    print(f"Qwen   EM: {qwen_trunc_em:.3f}   F1: {qwen_trunc_f1:.3f}")
    print(f"Wall clock: {t_trunc:.2f}s for {len(prompts)} examples")

    print("\n— 2-LLM joint (rescored pick on latent runs)")
    print(f"Joint  EM: {joint_em:.3f}  F1: {joint_f1:.3f}")
    print(f"Inter-model agreement (normalized): {agreement_rate:.3f}")
    print(f"Oracle upper bound:  EM {oracle_em:.3f}  F1 {oracle_f1:.3f}")

    # Machine-readable summary
    summary = {
        "samples": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "latent_len": int(latent_len),
        "avg_prompt_tokens": {"llama": avg_prompt_tokens_llama, "qwen": avg_prompt_tokens_qwen},
        "compression": {"llama": avg_prompt_tokens_llama/latent_len, "qwen": avg_prompt_tokens_qwen/latent_len},
        "payload_bytes": int(bytes_per_latent),
        "text": {
            "llama": {"em": llama_text_em, "f1": llama_text_f1, "nll_token": llama_text_nll},
            "qwen":  {"em": qwen_text_em,  "f1": qwen_text_f1,  "nll_token": qwen_text_nll},
            "wall_clock_sec": t_text,
        },
        "latent": {
            "llama": {"em": llama_latent_em, "f1": llama_latent_f1, "nll_token": llama_latent_nll},
            "qwen":  {"em": qwen_latent_em,  "f1": qwen_latent_f1,  "nll_token": qwen_latent_nll},
            "wall_clock_sec": t_latent,
        },
        "token_budget": {
            "llama": {"em": llama_trunc_em, "f1": llama_trunc_f1},
            "qwen":  {"em": qwen_trunc_em,  "f1": qwen_trunc_f1},
            "wall_clock_sec": t_trunc,
        },
        "joint": {
            "em": joint_em, "f1": joint_f1, "agreement": agreement_rate,
            "oracle": {"em": oracle_em, "f1": oracle_f1},
        },
    }

    import json as _json, os as _os
    print("\n==== METRICS_JSON ====")
    print(_json.dumps(summary, indent=2))
    if args.out_dir:
        _os.makedirs(args.out_dir, exist_ok=True)
        with open(_os.path.join(args.out_dir, "metrics.json"), "w") as f:
            _json.dump(summary, f, indent=2)
        # Simple CSV
        import csv as _csv
        with open(_os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["group","model","EM","F1","NLL/token","wall_clock_sec","compression","payload_bytes","samples","M"])
            w.writerow(["text","llama", summary["text"]["llama"]["em"], summary["text"]["llama"]["f1"], summary["text"]["llama"]["nll_token"], summary["text"]["wall_clock_sec"], summary["compression"]["llama"], summary["payload_bytes"], summary["samples"], summary["latent_len"]])
            w.writerow(["text","qwen",  summary["text"]["qwen"]["em"],  summary["text"]["qwen"]["f1"],  summary["text"]["qwen"]["nll_token"],  summary["text"]["wall_clock_sec"], summary["compression"]["qwen"],  summary["payload_bytes"], summary["samples"], summary["latent_len"]])
            w.writerow(["latent","llama", summary["latent"]["llama"]["em"], summary["latent"]["llama"]["f1"], summary["latent"]["llama"]["nll_token"], summary["latent"]["wall_clock_sec"], summary["compression"]["llama"], summary["payload_bytes"], summary["samples"], summary["latent_len"]])
            w.writerow(["latent","qwen",  summary["latent"]["qwen"]["em"],  summary["latent"]["qwen"]["f1"],  summary["latent"]["qwen"]["nll_token"],  summary["latent"]["wall_clock_sec"], summary["compression"]["qwen"],  summary["payload_bytes"], summary["samples"], summary["latent_len"]])
            w.writerow(["token_budget","llama", summary["token_budget"]["llama"]["em"], summary["token_budget"]["llama"]["f1"], "", summary["token_budget"]["wall_clock_sec"], summary["compression"]["llama"], summary["payload_bytes"], summary["samples"], summary["latent_len"]])
            w.writerow(["token_budget","qwen",  summary["token_budget"]["qwen"]["em"],  summary["token_budget"]["qwen"]["f1"],  "", summary["token_budget"]["wall_clock_sec"], summary["compression"]["qwen"],  summary["payload_bytes"], summary["samples"], summary["latent_len"]])
            w.writerow(["joint","both", summary["joint"]["em"], summary["joint"]["f1"], "", "", "", summary["payload_bytes"], summary["samples"], summary["latent_len"]])


if __name__ == "__main__":
    main()
