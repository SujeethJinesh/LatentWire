# Competitor Gap Analysis: C2C, KVComm, KVPress, KVzip, Quest

Date: 2026-04-21

Scope: current local competitor/reference state under `references/repos/` plus existing run artifacts under `results/` and `.debug/`. No new inference was run for this memo.

## Bottom Line

- `C2C` is the strongest already-run external communication baseline on the exact heterogeneous pair we care about: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`.
- The newly executed C2C GSM30 smoke on `data/gsm8k_gate_search_30.jsonl` ties target-alone at `0.0667` and trails the strict stochastic selector at `0.1667`; keep this as a small-slice smoke, not a final paper claim.
- `KVPress` is fully runnable locally and has completed same-model smoke checks, but the current evidence is only parity/tie on small probes.
- `KVComm` is runnable and ported once already on the Qwen pair, but the current ported GSM70 result is weak and the upstream code is natively same-family communication.
- `KVzip` and `Quest` are cloned and have runnable entrypoints, but I did not find local output artifacts for either in `results/`.

## What Is Already Runnable

### `C2C`

- Local repo clone: `references/repos/C2C`
- Bootstrap evidence: `results/c2c_bootstrap_20260418/summary.md`, `results/c2c_bootstrap_20260418/qwen_pair.json`
- Evaluation entrypoint: `references/repos/C2C/script/evaluation/unified_evaluator.py`
- Package/setup anchors: `references/repos/C2C/pyproject.toml`, `references/repos/C2C/environment.yml`

Status:

- Published Qwen pair is resolved and cached locally.
- Local import smoke passed.
- This is the only competitor here with strong exact-pair held-out GSM evidence already on disk.

### `KVComm`

- Local repo clone: `references/repos/KVComm`
- Entrypoints: `references/repos/KVComm/com.py`, `references/repos/KVComm/eval.py`
- Requirements anchor: `references/repos/KVComm/requirements.txt`

Status:

- The code is runnable and already ported once to the Qwen pair in our local harness.
- Upstream README is written for same-architecture communication tasks, so heterogeneous Qwen pair comparisons need explicit caveats.

### `KVPress`

- Local repo clone: `references/repos/kvpress`
- Entrypoints: `references/repos/kvpress/evaluation/evaluate.py`, `references/repos/kvpress/evaluation/evaluate_registry.py`
- Supported presses include `no_press`, `expected_attention`, `snapkv`, `streaming_llm`, `tova`, `observed_attention`, `qfilter`, `pyramidkv`, `kvzip`, and others.

Status:

- Native benchmark runner is available locally.
- The repo supports the benchmark families we need for paper-safe controls: `needle_in_haystack`, `ruler`, `longbench`, `loogle`, `zero_scrolls`, `math500`, and `aime25`.

### `KVzip`

- Local repo clone: `references/repos/KVzip`
- Entrypoints: `references/repos/KVzip/test.py`, `references/repos/KVzip/eval.py`, `references/repos/KVzip/results/parse.py`
- Requirements anchor: `references/repos/KVzip/requirements.txt`

Status:

- The repo is present and runnable in principle.
- I found no local result artifacts for KVzip under `results/`.

### `Quest`

- Local repo clone: `references/repos/Quest`
- Entrypoints: `references/repos/Quest/scripts/passkey.sh`, `references/repos/Quest/scripts/longbench.sh`, `references/repos/Quest/scripts/ppl_eval.sh`, `references/repos/Quest/evaluation/LongBench/eval.py`

Status:

- The repo is present and runnable in principle.
- I found no local result artifacts for Quest under `results/`.
- README targets Llama/Mistral-family long-context setups and CUDA-heavy kernels, so it is not the smallest local smoke path.

## Existing Results And Memo Anchors

### `C2C` results

- `results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl.meta.json`
- `results/c2c_gsm70_20260418/summary.md`
- `results/c2c_gsm100_20260418/qwen_gsm100_c2c.jsonl.meta.json`
- `results/c2c_gsm100_20260418/summary.md`
- `results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl.meta.json`
- `results/c2c_bootstrap_20260418/summary.md`
- `results/competitor_bootstrap_20260421/c2c_qwen_gsm5_native_20260421.jsonl.meta.json`
- `results/competitor_bootstrap_20260421/c2c_qwen_gsm30_native_20260421.jsonl.meta.json`

Observed values:

- GSM70 accuracy: `0.12857142857142856`
- GSM100 accuracy: `0.11`
- SVAMP70 accuracy: `0.44285714285714284`
- GSM5 native smoke accuracy: `0.0`
- GSM30 native smoke accuracy: `0.06666666666666667`

### `KVPress` results

- `.debug/kvpress_nih/needle_in_haystack__Qwen--Qwen3-0.6B__no_press__0.00__max_context4096__query_aware__needle_depth50/metrics.json`
- `.debug/kvpress_nih/needle_in_haystack__Qwen--Qwen3-0.6B__expected_attention__0.50__max_context4096__query_aware__needle_depth50/metrics.json`
- `results/kvpress_expected_20260420/qwen_gsm5_no_press.jsonl.meta.json`
- `results/kvpress_expected_20260420/qwen_gsm5_expected_attention.jsonl.meta.json`
- `results/tmp_kvpress_none_gsm5.jsonl.meta.json`
- `results/tmp_kvpress_expected_gsm5.jsonl.meta.json`
- Prior planning notes: `references/295_competitor_bootstrap_triage.md`, `references/301_competitor_benchmark_bootstrap.md`, `references/302_competitor_next_benchmarks.md`, `references/303_competitor_quant_cache_benchmarks.md`, `references/300_telemetry_blocker_synthesis.md`

Observed values:

- GSM5 wrapper accuracy: `0.2` for both `none` and `expected_attention`
- Native needle ROUGE-L F: `0.75` for both `no_press` and `expected_attention`

### `KVComm` results

- `results/kvcomm_gsm70_20260419/qwen_gsm70_kvcomm_ported.jsonl.meta.json`

Observed values:

- GSM70 ported accuracy: `0.0`
- Best calibrated layer fraction in that run: `0.5`

### `KVzip` / `Quest` results

- I did not find local result artifacts under `results/` for either repo.
- For KVzip, only code-side helpers exist in `references/repos/KVzip/results/`.

## What Is Fair Vs Unfair To Compare

### Fair

- Same benchmark family, same split, same parser, same decoding budget, same tokenizer/model pair.
- `KVPress` within the same family only:
  - GSM wrapper vs GSM wrapper
  - native `needle_in_haystack` vs native `needle_in_haystack`
  - native `ruler` vs native `ruler`
- `C2C` on the exact same source/target pair and held-out GSM split.
- `KVComm` only as a ported cross-model communication control on the same pair and split we use elsewhere.
- `KVzip` only as a same-model compression control.
- `Quest` only as a query-aware sparsity control once the model family and CUDA/runtime match the repo assumptions.

### Unfair

- Comparing GSM exact-match accuracy to needle ROUGE, LongBench F1/ROUGE, or SVAMP as if they are interchangeable.
- Comparing same-model compression baselines (`KVPress`, `KVzip`, `Quest`) directly to cross-model communication baselines (`C2C`, `KVComm`) without labeling the task difference.
- Comparing native `KVComm` same-family claims to our heterogeneous Qwen pair without saying that the pair was adapted.
- Mixing throughput/latency with accuracy claims.
- Treating a tiny smoke as proof of a paper-level win.

## Smallest Paper-Safe Next Runs

1. `KVPress` native replay on a tiny retrieval benchmark, preferably `needle_in_haystack` or `ruler`, with `no_press` vs `expected_attention` held constant on the same `model`, `fraction`, `compression_ratio`, `query_aware`, `max_context_length`, and `needle_depth`.
2. `KVzip` same-model smoke on a minimal slice of a supported dataset such as `squad` or `scbench_kv`, using `references/repos/KVzip/test.py` first so we can verify the harness before any wider sweep.
3. `KVComm` tiny GSM port on the exact same heterogeneous Qwen pair, using a 5- or 10-example slice so we can tell whether the current `0.0` GSM70 port is a calibration miss or a true ceiling.
4. `C2C` parser-matched scale-up on `gsm8k_eval_70` only after confirming the native-vs-wrapper parser path is identical to the current table.

## Recommended Paper Framing

- Treat `C2C` as the current external communication anchor on the exact Qwen pair.
- Treat `KVPress` and `KVzip` as compression controls, not communication wins.
- Treat `Quest` as a CUDA-heavy sparsity control that is useful later, but not the smallest local comparator right now.
- Use the exact file paths above in the paper notes so the benchmark lineage stays auditable.
