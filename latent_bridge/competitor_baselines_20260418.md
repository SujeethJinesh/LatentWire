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
