import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer

def test_end_to_end_smoke():
    mA = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    mB = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

    device = "cpu"
    dtype = torch.float32

    A = LMWrapper(LMConfig(model_id=mA, dtype=dtype))
    B = LMWrapper(LMConfig(model_id=mB, dtype=dtype))

    enc = InterlinguaEncoder(d_z=64, latent_len=4).to(device)
    adpA = Adapter(d_z=64, d_model=A.d_model).to(device)
    adpB = Adapter(d_z=64, d_model=B.d_model).to(device)

    bt = ByteTokenizer(max_bytes=128)
    texts = ["Question: 2+2?\nContext: math.\nAnswer:", "Question: Capital of France?\nContext: eu.\nAnswer:"]
    answers = ["4", "Paris"]

    ids = [bt.encode(t) for t in texts]
    maxT = max(x.numel() for x in ids)
    batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.numel(), dtype=torch.long)], 0) for x in ids], 0)

    with torch.no_grad():
        Z = enc(batch)                  # [2, 4, 64]
        pA = adpA(Z)                    # [2, 4, dA]
        pB = adpB(Z)                    # [2, 4, dB]
        outA = A.generate_from_prefix(pA, max_new_tokens=8)
        outB = B.generate_from_prefix(pB, max_new_tokens=8)

    assert len(outA) == 2 and len(outB) == 2
    sA = [A.tokenizer.decode(o, skip_special_tokens=True) for o in outA]
    sB = [B.tokenizer.decode(o, skip_special_tokens=True) for o in outB]
    assert all(isinstance(x, str) and len(x) >= 0 for x in sA+sB)
