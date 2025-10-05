
# train.py (anchor tokens): build anchor mask + keep only anchors' KV at eval-like decode
import argparse, torch, torch.nn.functional as F
from models import LMWrapper
from losses import first_token_ce, k_token_ce, kd_first_k
from core_utils import build_anchor_mask, select_anchor_kv
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--first_token_ce_weight", type=float, default=8.0)
    ap.add_argument("--k_ce_weight", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--kd_first_k_weight", type=float, default=1.0)
    ap.add_argument("--kd_tau", type=float, default=2.0)
    ap.add_argument("--anchor_window", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = LMWrapper(args.model_id).to(device)
    tok = lm.tokenizer
    optim = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=args.lr)

    for step in range(10):
        messages = [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"Question: Who wrote Hamlet?\nContext: Shakespeare wrote it.\n"}
        ]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        labels = enc["input_ids"].clone()
        B,T = labels.shape
        # mark anchors at fixed window and assistant start
        is_anchor = torch.zeros(B,T, dtype=torch.bool, device=device)
        is_anchor[:, ::args.anchor_window] = True
        is_anchor[:, 0] = True

        out = lm(**enc, labels=labels, return_dict=True)
        logits = out.logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=tok.pad_token_id or -100)
        loss += args.first_token_ce_weight * first_token_ce(logits, labels)
        loss += args.k_ce_weight * k_token_ce(logits, labels, K=args.K)
        with lm.disable_adapters():
            loss += args.kd_first_k_weight * kd_first_k(lm.model, lm.model, enc["input_ids"], enc.get("attention_mask"),
                                                        labels, K=args.K, tau=args.kd_tau)

        optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); optim.step()
        print(f"step {step} loss={loss.item():.4f}")

if __name__ == "__main__":
    main()
