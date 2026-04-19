## Competitor Baselines

Cloned repos:

- `references/repos/C2C`
- `references/repos/KVComm`
- `references/repos/LatentMAS`

### First fair comparison

`C2C` is the first competitor baseline to prioritize.

Why:

- its README exposes a published checkpoint for the exact pair
  `Qwen/Qwen3-0.6B` + `Qwen/Qwen2.5-0.5B-Instruct`
- the Hugging Face bundle is reachable from this machine at
  `qwen3_0.6b+qwen2.5_0.5b_Fuser/config.json`, so the baseline path is real and
  not just mentioned in the README
- it already has a generation evaluator for GSM8K-style tasks
- it is the closest direct baseline to our current same-pair GSM story

Primary blocker:

- their evaluator and prompt formatting are separate from our JSONL split, so
  the first integration step is a replay on our held-out GSM slices rather than
  their stock config

### Secondary baselines

`KVComm` is the next cleanest control because it is training-free, but it is
less turnkey for our current GSM JSONL path and is better treated as the second
comparison after `C2C`.

Current blocker on `KVComm`:

- the checked-in repo is not Qwen3-ready; its wrappers special-case `Llama`,
  `Qwen2`, and `Gemma3`, so `Qwen/Qwen3-0.6B` is a real compatibility patch,
  not a drop-in replay
- its evaluator expects its own task format, so our GSM JSONL split still
  needs a thin adapter before it becomes a fair apples-to-apples comparison
- the exact Qwen pair also mismatches both KV-head count and per-head
  dimensionality (`2 -> 8` KV heads and `64 -> 128` head dim), so stock
  selective KV sharing is not directly executable on this heterogeneous pair
  without an explicit compatibility lift

`LatentMAS` is useful for broader latent-collaboration context, but it is not
the cleanest direct protocol match for our current pairwise sparse-K routing
story.

### Immediate next steps

1. bootstrap `C2C` on the exact Qwen pair with its published checkpoint
2. replay it on `data/gsm8k_eval_70.jsonl` or `data/gsm8k_100.jsonl`
3. compare against:
   - target-alone
   - text-to-text
   - zero-byte attenuation
   - our best current sparse routing branch

Bootstrap command:

```bash
python scripts/bootstrap_c2c.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --download
```

Replay command:

```bash
python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device mps \
  --max-new-tokens 64 \
  --prediction-output results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl
```

Current read:

- `C2C` on `gsm8k_eval_70`: `0.128571`
- current best same-pair internal branch on the same split: `0.085714`
- `C2C` on `gsm8k_100`: `0.110000`
- current best same-pair internal branch on the same larger slice: `0.070000`
- `C2C` on `svamp_eval_70`: `0.442857`
- current best same-pair internal SVAMP branch on the same split: `0.171429`
- older text-to-text reference on the same SVAMP split: `0.414286`
- stock `KVComm` is not natively runnable on the same Qwen pair because of the
  geometry mismatch above
- a compatibility-lifted `KVComm` replay on `gsm8k_eval_70` still scored
  `0.000000`
- the held-out dev sweep for that lifted replay peaked only at `0.033333` on
  `gsm8k_gate_search_30` with a `0.50` top-layer fraction
- paired against our best internal GSM70 branch, lifted `KVComm` is
  `-0.085714` with `0` KVComm-only wins and `6` internal-only wins
- paired against `C2C`, lifted `KVComm` is `-0.128571` with `0` KVComm-only
  wins and `9` C2C-only wins
