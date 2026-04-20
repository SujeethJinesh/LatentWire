## Competitor Baselines

Cloned repos:

- `references/repos/C2C`
- `references/repos/DeltaKV_sparse_vllm`
- `references/repos/KVComm`
- `references/repos/KVzip`
- `references/repos/Quest`
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

`DeltaKV` is now the next highest-value external control after `C2C`.

Why:

- it is the strongest newer **residual-style compression control** we found
  with public code and is now cloned locally at
  `references/repos/DeltaKV_sparse_vllm`
- it is the closest already-public control to our recent “frozen base plus
  correction” family, even though it remains a compression/reconstruction
  baseline rather than a heterogeneous transport method
- if we spend another comparator day, it is the cleanest next test of whether
  our transport-plus-correction story is really different from modern
  residual-view KV compression

Current blocker on `DeltaKV`:

- it is not a direct heterogeneous communication baseline, so it belongs in
  the paper as an external control rather than an apples-to-apples peer method
- it still needs a thin replay harness on our GSM JSONL slices before it
  becomes a fair table row

`KVzip` is now the next already-cloned fallback control after `DeltaKV`.

Why:

- it is the strongest modern **compression-side control** for separating
  cross-model communication from generic cache compression / reconstruction
- it has public code and is now cloned locally at `references/repos/KVzip`
- it remains a better next already-cloned comparator day than `KVComm`

Current blocker on `KVzip`:

- it is a cache-compression control rather than a cross-model transport method,
  so the paper should present it as an external control, not an apples-to-
  apples heterogeneous communication baseline
- its stock path still needs a thin replay harness on our GSM JSONL slices to
  become a fair direct table row

`Quest` is the next cleanest query-aware sparsity control after `KVzip`.

Why:

- it is the strongest older query-aware pruning control and is now cloned
  locally at `references/repos/Quest`
- it is useful if we want to rule out “any query-aware KV pruning works” as a
  confound for our method family

Current blocker on `Quest`:

- it is a long-context inference method, not a heterogeneous communication
  baseline
- its evaluation path still needs a thin adapter to replay on our GSM JSONL
  splits

`KVComm` is still useful, but it has dropped below `KVzip` and `Quest` in
immediate comparator value.

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

`DapQ` is the strongest newer decoding-aligned control we found, but it does
not yet have a clean public repo path to clone and replay.

Why it matters:

- it is the cleanest recent *decoding-aligned* control for the same general
  question of query-aware KV selection
- if a stable repo appears, it could leapfrog `KVzip` as the best mechanistic
  control after `DeltaKV`

Latest read on `LatentMAS`:

- the public examples center on `Qwen/Qwen3-14B` and sequential multi-agent
  collaboration rather than direct heterogeneous source-target transport
- it is better treated as a broader latent-collaboration context citation than
  as the next fair apples-to-apples baseline for the current
  `Qwen2.5-0.5B -> Qwen3-0.6B` pair
- `KVComm` remains the faster adjacent systems-style comparator to add after
  `C2C`, while `LatentMAS` is higher risk as a direct replay

### Immediate next steps

1. keep `C2C` as the main external bar
2. bootstrap `DeltaKV` next as the highest-value external control day
3. keep `KVzip` as the next already-cloned fallback control
4. keep `Quest` as the next pruning-style fallback control
4. compare against:
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

### Exact KVPress / Expected Attention

The vendored `KVPress` baseline is now runnable exactly on our local slices
through `scripts/run_kvpress_eval.py`, which patches the clone's cache API
compatibility at runtime rather than relying on manual edits inside
`references/repos/kvpress`.

Exact held-out reads on `Qwen/Qwen3-0.6B` with shared-chat prompts and
`enable_thinking=False`:

- `data/gsm8k_5.jsonl`, no press: `0.200000`
- `data/gsm8k_5.jsonl`, `ExpectedAttentionPress`, `compression_ratio=0.5`:
  `0.200000`
- `/tmp/gsm8k_eval_10.jsonl`, no press: `0.100000`
- `/tmp/gsm8k_eval_10.jsonl`, `ExpectedAttentionPress`,
  `compression_ratio=0.5`: `0.100000`

Saved artifacts:

- `results/kvpress_expected_20260420/qwen_gsm5_no_press.jsonl.meta.json`
- `results/kvpress_expected_20260420/qwen_gsm5_expected_attention.jsonl.meta.json`
- `results/kvpress_expected_20260420/qwen_gsm10_no_press.jsonl.meta.json`
- `results/kvpress_expected_20260420/qwen_gsm10_expected_attention.jsonl.meta.json`

Interpretation:

- exact external Expected Attention matches its own no-press floor on both
  slices
- this makes KVPress / Expected Attention an honest **negative-boundary
  comparator** on our current pair, not a live positive baseline
- `C2C` remains the main external bar; KVPress is useful as a compression-side
  boundary and calibration sanity check
- immediate comparator priority is now:
  1. `C2C` as the main bar
  2. `DeltaKV` as the next residual-style compression control
  3. `DapQ` as the next decoding-aligned query-aware control if the public
     code path becomes practical
  4. `KVzip` as the next already-cloned compression-side control
  5. `Quest` as the next query-aware pruning control
  6. `KVComm` as a lower-priority adjacent replay
