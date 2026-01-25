# Implementation Plan + Experiment Matrix (1x H100 per milestone)

Goal: deliver a workshop‑ready paper on **Quantized Cache‑to‑Cache** with communication‑budget curves, and a clear path to a main‑conference submission (QAT + mixed precision + heterogeneity scaling). Each milestone is designed to run on a single H100.

## Data Capture Contract (applies to every milestone)
- Each run creates `quantization/data/step_X_<name>/<run_tag>/` with (folder names remain `step_X` for now to match scripts):
  - `configs/` (the exact eval/train configs used)
  - `logs/` (stdout/stderr logs)
  - `results/` (JSON outputs)
  - `manifests/` (checkpoint + environment provenance)
- Commit `quantization/data/step_*` to git after each milestone so results are reviewable.
- Cleanup hygiene: remove failed runs that do not contain `results/` or have `status=failed` in manifests (keep only validated runs).

## C2C Fork Strategy (for Milestone 1+ changes)
**Decision**  
Fork C2C and point the submodule at the fork. Keep all quantization changes on a dedicated branch (e.g., `quant-kv`).

**Why**  
This is the cleanest path for reproducibility and publication: exact diffs are visible, and upstream sync is straightforward.

**How**  
- Create fork on GitHub, add `upstream` remote for the original repo.  
- Update submodule remote to the fork and pin to the branch.  
- Keep changes minimal and well‑scoped (quantization utilities + hook points only).

## Milestone 0: Baseline + Environment Sanity
**What**  
Run the official C2C baselines on OpenBookQA and ARC‑C with the released 0.6B+0.5B fuser.

**Why**  
Establish a trustworthy baseline before introducing quantization or cache‑budget changes.

**How**  
Use `quantization/scripts/run_step0_baselines.py` (supports `--prep-only` on login nodes).  
The script:
- Ensures the C2C submodule is initialized.
- Verifies the `rosetta` conda env and installs deps if missing.
- Downloads the published fuser checkpoint to scratch.
- Writes `env_info.json` and checkpoint manifests.
- Runs OpenBookQA + ARC‑C and logs everything into `quantization/data/step_0_baselines/<run_tag>/`.

**Commands (from repo root)**  
- Login node prep: `python quantization/scripts/run_step0_baselines.py --prep-only`  
- GPU node eval: `python quantization/scripts/run_step0_baselines.py`

**Workshop/Main‑conf connection**  
Baseline accuracy and latency anchors the whole paper.

## Milestone 1: Implement KV PTQ Utilities
**What**  
Add post‑training quantization (INT8, INT4/NF4, optional FP8) for KV caches.

**Why**  
Quantization is not covered in C2C; it is the core novelty for the workshop paper.

**How**  
Add a small quantization module (e.g., `rosetta/utils/quant.py`) and hook it before projection in `rosetta/model/wrapper.py`.  
Expose a `kv_quant_config` block (per‑head / per‑layer) under `model.rosetta_config`:  
```yaml
kv_quant_config:
  enabled: true
  scheme: int8   # int8 | int4
  axis: head     # head | layer
  eps: 1e-6
  collect_stats: false
```  
Follow the Milestone 0 pattern with a minimal, well‑scoped runner:
- `quantization/scripts/run_step1_kv_ptq.py` with GPU as default; `--mode local` for Mac validation.  
- Local smoke test: **one real sample** from OpenBookQA or ARC‑C using the same C2C eval template + `extract_answer_from_content` check.  
- GPU eval: identical to Step 0, but with quantization flags enabled.  

**Workshop/Main‑conf connection**  
PTQ results + bandwidth reductions are sufficient for a workshop paper.

**Commands (from repo root)**  
- Login node prep (INT8): `python quantization/scripts/run_step1_kv_ptq.py --prep-only --kv-quant-scheme int8`  
- GPU node eval (INT8): `python quantization/scripts/run_step1_kv_ptq.py --kv-quant-scheme int8`  
- GPU node eval (INT4): `python quantization/scripts/run_step1_kv_ptq.py --kv-quant-scheme int4`  
- Local dry run (1 sample): `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --kv-quant-scheme int8`  
- Local mini‑batch (5 samples): `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 5 --kv-quant-scheme int8`  
- Local no‑quant baseline (5 samples): `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 5 --disable-kv-quant`  
- Cleanup failed runs: add `--cleanup-failed` to remove incomplete folders.

**Milestone 1 sub‑steps (phased)**  
- **Phase 0 (Plan update)**: Document decisions, sub‑steps, and extensions (this section).  
- **Phase 1 (Fork changes)**: Add `rosetta/utils/quant.py` + a single hook in `rosetta/model/wrapper.py` for source‑KV quantization.  
- **Phase 2 (Runner)**: Add `quantization/scripts/run_step1_kv_ptq.py` mirroring Step 0, with GPU default and local dry‑run support.  
- **Phase 3 (Local validation)**: Run a **single dataset sample** (OpenBookQA or ARC‑C) to validate prompt formatting, quantization path, and logging.  
- **Phase 4 (GPU eval)**: Run full OpenBookQA + ARC‑C with quantization enabled; compare accuracy vs Step 0.  

**Milestone 1 Phase 1 implementation details (applied)**  
- **Quantization module**: `rosetta/utils/quant.py` implements symmetric fake‑quant for KV (INT8/INT4), per‑head or per‑layer.  
- **Hook point**: `rosetta/model/wrapper.py` quantizes **source KV slices** immediately before projection.  
- **Config plumbing**: `rosetta/utils/evaluate.py` passes `kv_quant_config` into `RosettaModel`.  
- **Design choice**: fake‑quant only (quantize → dequantize), no bit‑packing yet. This preserves execution path while modeling quantization noise.
- **Interpretability**: optional `collect_stats` aggregates min/max/mean scale stats per run (low overhead when enabled).

**Decisions + justification**  
- **Start with INT8 + INT4 only**: These are the most standard, hardware‑agnostic PTQ baselines. They deliver a clear accuracy/bytes trade‑off and are sufficient for a workshop paper.  
  - **Extensions (later)**: NF4 and FP8. NF4 can improve INT4 accuracy; FP8 is H100‑friendly and could strengthen a main‑conf submission.  
- **Quantize source KV only**: This models the actual communication channel (teacher → base) and isolates the quantization effect to the transmitted cache.  
  - **Extensions (later)**: quantize base KV, quantize both source+base, or quantize only a subset of layers.  
- **Default per‑head axis**: Per‑head scales typically preserve accuracy better than per‑layer with modest overhead.  
  - **Extensions (later)**: per‑layer (cheaper), per‑token (more accurate but expensive), or mixed granularity by layer depth.  

**Extensions to Milestone 1 (backburner)**  
- **NF4 / FP8**: add additional schemes for improved INT4 accuracy or H100‑optimized precision.  
- **Bit‑packing**: actual KV compression for realistic bandwidth measurements.  
- **Source+base quantization**: quantify whether quantization noise on both sides compounds or cancels.  
- **Layer‑selective quantization**: higher precision in late layers, lower in early layers.  
- **Group‑wise quantization**: group size tuning for accuracy/overhead trade‑offs.  
- **Per‑run scale histograms**: richer quantization diagnostics beyond min/max/mean.  

**Technical notes / acronyms (Milestone 1)**  
- **KV cache**: Attention Keys/Values stored for past tokens to avoid recomputation.  
- **C2C**: Cache‑to‑Cache projection from a source model’s KV to a target model’s KV.  
- **PTQ**: Post‑Training Quantization; quantize activations without retraining.  
- **INT8/INT4**: 8‑bit / 4‑bit integer quantization (smaller = less bandwidth).  
- **NF4**: Normalized 4‑bit quantization (non‑uniform 4‑bit, often higher accuracy than plain INT4).  
- **FP8**: 8‑bit floating point (hardware‑dependent, often H100‑friendly).  
- **Per‑head vs per‑layer**: granularity of scales/zero‑points; per‑head is higher accuracy, per‑layer is cheaper.  

## Milestone 2: PTQ Evaluation
**What**  
Evaluate INT8 and INT4/NF4 on OpenBookQA + ARC‑C.

**Why**  
Quantifies accuracy drop vs bandwidth savings.

**How**  
Run the same eval pipeline as Step 0 with quantization flags enabled.  
Store results under `quantization/data/step_1_kv_ptq/<run_tag>/` for now (rename to `step_2_ptq` if we split a dedicated runner later).

**Milestone 2 sub‑steps (phased)**  
- **Phase 0 (Plan update)**: lock down eval datasets (OpenBookQA + ARC‑C), quant schemes (INT8/INT4), and the storage contract.  
  - **Why**: avoids drifting benchmarks and ensures deltas are attributable to quantization only.  
- **Phase 1 (Config + runner)**: ensure Step 1 runner writes eval configs with `kv_quant_config` injected and output dirs scoped per run.  
  - **Why**: keeps eval settings identical to Milestone 0 while toggling only KV precision.  
- **Phase 2 (Local sanity)**: run 5–20 local samples with quant enabled vs `--disable-kv-quant` to confirm quantization does not change behavior in small samples.  
  - **Why**: verifies the quant path is exercised and stable before burning GPU time.  
  - **Notes**: keep `include_response=false` (default) to match paper evals; record answer distribution (A/B/C/D) to spot degenerate outputs.  
- **Phase 3 (Local receiver‑only baseline)**: run the receiver model without C2C projectors on the same samples.  
  - **Why**: isolates base model behavior and helps explain degenerate outputs before GPU access.  
- **Deferred GPU block (execute when GPU is available)**  
  - **Phase 4 (GPU smoke)**: run a limited GPU eval (e.g., 25–50 samples) by temporarily setting `eval.limit` in the generated configs to validate throughput + correctness.  
    - **Why**: catches runtime errors and logging issues quickly on the real target environment.  
    - **Notes**: verify answer distribution and output length stats before running full eval.  
  - **Phase 5 (Receiver‑only sanity)**: run a short receiver‑only baseline on the same subset with the same prompt settings to confirm outputs are not degenerate (sanity check vs Table 3).  
    - **Why**: ensures the base model is behaving reasonably before we attribute deltas to C2C or quantization.  
  - **Phase 6 (Full GPU eval)**: run full OpenBookQA + ARC‑C for INT8 and INT4; capture results + logs.  
    - **Why**: these numbers anchor the workshop paper and the budget‑accuracy curves.  
  - **Phase 7 (Analysis + QC)**: compute delta accuracy vs Milestone 0, check consistency with C2C paper settings.  
    - **Why**: prevents accidental misalignment with the baseline or paper protocol.

**Telemetry (Milestone 2)**  
- `env_info.json`: conda, HF cache paths, torch/transformers versions, CUDA visibility.  
- `manifests/step_1_manifest.json`: checkpoint + git SHA provenance and `kv_quant_config`.  
- `logs/step1.log`: full evaluator stdout/stderr for reproducibility.  
- `results/`: evaluator JSON outputs per dataset.  
- Optional: `kv_quant_stats` saved by local runs to confirm quant path is exercised.
- Add a small analysis note after each GPU run: answer distribution (A/B/C/D counts) and mean output length to catch degenerate outputs early.

**Paper‑alignment checklist (pre‑GPU)**  
- `include_response=false` (default in C2C eval config).  
- `use_template=true`, `use_cot=false`, `max_new_tokens=64`, temperature = 0.  
- Datasets: OpenBookQA + ARC‑C test splits; zero‑shot prompts.  
- Model pair: Qwen3‑0.6B (receiver) + Qwen2.5‑0.5B (sharer), matching Table 3.  
- `is_do_alignment=false` unless explicitly testing alignment.

**Local test commands (Milestone 2)**  
- Quant vs no‑quant (OpenBookQA, 20 samples):  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 20 --kv-quant-scheme int8 --kv-quant-collect-stats`  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 20 --disable-kv-quant`  
- Receiver‑only local baseline (OpenBookQA, 20 samples):  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 20 --local-receiver-only`  

**GPU smoke commands (Milestone 2, deferred)**  
- Prep + config patch (INT8 example, repeat for INT4):  
  - `export PROJECT_ROOT=/projects/m000066/sujinesh/LatentWire`  
  - `export RUN_TAG=int8_smoke_$(date +%Y%m%d_%H%M%S)`  
  - `export C2C_CKPT_ROOT=/scratch/m000066/$USER/c2c_checkpoints`  
  - `export HF_HOME=/scratch/m000066/$USER/.cache/huggingface`  
  - `cd $PROJECT_ROOT`  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode gpu --kv-quant-scheme int8 --prep-only --run-tag "$RUN_TAG"`  
  - `python - <<'PY'\nimport sys\nfrom pathlib import Path\nimport yaml\nrun_tag = sys.argv[1]\nrun_root = Path('quantization/data/step_1_kv_ptq') / run_tag\nfor name in ('openbookqa.yaml', 'arc_c.yaml'):\n    path = run_root / 'configs' / name\n    cfg = yaml.safe_load(path.read_text())\n    cfg.setdefault('eval', {})['limit'] = 50\n    path.write_text(yaml.safe_dump(cfg, sort_keys=False))\nPY "$RUN_TAG"`  
- Run evaluator with logging (use the run’s log file):  
  - `python quantization/C2C/script/evaluation/unified_evaluator.py --config quantization/data/step_1_kv_ptq/$RUN_TAG/configs/openbookqa.yaml 2>&1 | tee -a quantization/data/step_1_kv_ptq/$RUN_TAG/logs/step1.log`  
  - `python quantization/C2C/script/evaluation/unified_evaluator.py --config quantization/data/step_1_kv_ptq/$RUN_TAG/configs/arc_c.yaml 2>&1 | tee -a quantization/data/step_1_kv_ptq/$RUN_TAG/logs/step1.log`  

**Full GPU eval commands (Milestone 2, deferred)**  
- INT8 full run: `python quantization/scripts/run_step1_kv_ptq.py --mode gpu --kv-quant-scheme int8`  
- INT4 full run: `python quantization/scripts/run_step1_kv_ptq.py --mode gpu --kv-quant-scheme int4`  

## Milestone 3: Cache‑Length Reduction
**What**  
Prune KV tokens (e.g., keep top‑50%, 25%, 10%) before projection.

**Why**  
Adds a second “budget” axis (length) beyond precision, enabling stronger trade‑off curves.

**How**  
Use the existing `kv_cache_proportion` + `kv_cache_order_mode` hooks in the evaluator to mask a fraction of instruction tokens (simple front/back pruning) and evaluate with INT8.

**Milestone 3 sub‑steps (phased)**  
- **Phase 0 (Plan update)**: lock the cache‑length grid and defaults.  
  - **Default**: `kv_cache_proportion=1.0`, `kv_cache_order_mode=front`.  
  - **Grid**: {1.0, 0.75, 0.5, 0.25, 0.1} × {front, back}.  
  - **Why**: aligns with the “budget curve” goal while keeping the search small enough for 1×H100.  
- **Phase 1 (Runner update)**: extend the GPU runner to accept `--kv-cache-proportion` and `--kv-cache-order-mode` and inject them into the generated eval configs; add these fields to `step_1_manifest.json` and local summaries.  
  - **Why**: ensures all runs are traceable and reduces manual config edits.  
- **Phase 2 (Local sanity)**: run 5–20 local samples with `proportion=1.0` vs `0.5` (front) to validate the kv_cache_index masking path.  
  - **Why**: quick plumbing check before GPU time.  
- **Phase 3 (Local order‑mode check)**: run 5–20 local samples with `proportion=0.5` and `order_mode=back` to verify ordering is wired.  
  - **Why**: ensures we can separate “keep early” vs “keep late” effects.  
- **Deferred GPU block (execute when GPU is available)**  
  - **Phase 4 (GPU smoke)**: run a 25–50 sample limit for two settings (0.5/front, 0.5/back).  
    - **Why**: validate throughput + correctness under the real target environment.  
  - **Phase 5 (Full GPU grid)**: run OpenBookQA + ARC‑C for the full grid above using INT8 PTQ.  
    - **Why**: this yields the core “accuracy vs length” curve for the workshop.  
  - **Phase 6 (Analysis + QC)**: compute accuracy vs proportion and produce a single curve per dataset; verify output lengths and answer distributions.  
    - **Why**: catches degenerate outputs and produces paper‑ready plots.

**Telemetry (Milestone 3)**  
- `step_1_manifest.json`: include `kv_cache_proportion` and `kv_cache_order_mode`.  
- `results/*_summary.json`: record accuracy and length stats per setting.  
- `progress/*.json`: chunked resume state for long runs (GPU).  
- `logs/step1.log`: combined stdout/stderr for auditing and failures.

**Local test commands (Milestone 3)**  
- After Phase 1 is implemented, run:  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 20 --kv-quant-scheme int8 --kv-cache-proportion 1.0 --kv-cache-order-mode front`  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 20 --kv-quant-scheme int8 --kv-cache-proportion 0.5 --kv-cache-order-mode front`  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-sample-index 0 --local-num-samples 20 --kv-quant-scheme int8 --kv-cache-proportion 0.5 --kv-cache-order-mode back`  

**GPU commands (Milestone 3, deferred)**  
- Use the chunked SLURM runner, but override `kv_cache_proportion` and `kv_cache_order_mode` per run tag (after Phase 1 adds flags).  
- Start with smoke limits (50 samples) for 0.5/front and 0.5/back; then expand to the full grid.

## Milestone 4: Communication‑Budget Curves
**What**  
Report accuracy vs transmitted bytes (precision × KV length).

**Why**  
This reframes C2C under a realistic communication budget, a key paper contribution.

**How**  
Log bytes in evaluation outputs and produce a single “accuracy vs bytes” plot.

## Milestone 5: QAT Recovery (Main‑conf extension)
**What**  
Quantization‑aware training of the projector under INT8 noise.

**Why**  
Shows that low‑precision transfer can be learned, not just approximated.

**How**  
Train on a small subset (10–50k samples), then re‑evaluate ARC‑C + OpenBookQA.

## Milestone 6: Mixed Precision by Layer (Main‑conf extension)
**What**  
Use higher precision in later layers and lower precision in early layers.

**Why**  
Leverages layer sensitivity to improve the accuracy‑per‑byte curve.

**How**  
Add per‑layer precision configs; evaluate a small grid (e.g., last 4 layers FP16).

## Milestone 7: Heterogeneity Scaling (Main‑conf extension)
**What**  
Test at least one cross‑family pair (e.g., Qwen3 ← Llama3.2).

**Why**  
Demonstrates that the method generalizes beyond a single model family.

**How**  
Reuse the PTQ + pruning settings from Steps 2–3.

---

# Experiment Matrix (Merged)

## A. Baselines (Milestone 0)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| A1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C | FP16 | full | no |
| A2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C | FP16 | full | no |

## B. PTQ (Milestones 1–2)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| B1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C + PTQ | INT8 | full | no |
| B2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | full | no |
| B3 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C + PTQ | INT4/NF4 | full | no |
| B4 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT4/NF4 | full | no |

## C. Cache‑Length Reduction (Milestone 3)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| C1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | 50% | no |
| C2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | 25% | no |
| C3 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + PTQ | INT8 | 10% | no |

## D. QAT / Mixed Precision (Milestones 5–6)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| D1 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | C2C + QAT | INT8 | full | yes |
| D2 | Qwen3‑0.6B ← Qwen2.5‑0.5B | OpenBookQA | C2C + QAT | INT8 | full | yes |
| D3 | Qwen3‑0.6B ← Qwen2.5‑0.5B | ARC‑C | Mixed precision | FP16/INT8/INT4 | full | yes |

## E. Heterogeneity (Milestone 7)
| ID | Model Pair | Dataset | Method | Precision | Cache Length | Train |
|---|---|---|---|---|---|---|
| E1 | Qwen3‑0.6B ← Llama3.2‑1B | ARC‑C | C2C + PTQ | INT8 | full | no |
| E2 | Qwen3‑0.6B ← Gemma‑1B | ARC‑C | C2C + PTQ | INT8 | full | no |

---

## Workshop vs Main‑Conference Path
- **Workshop**: Milestones 0 → 1 → 2 → 3 → 4 (baseline + PTQ + pruning + budget curve).
- **Main‑conf**: Workshop path + Milestones 5 → 6 → 7 (QAT + mixed precision + heterogeneity).

## Potential Improvements / Follow‑ups
- **Byte accounting**: include per‑head scale metadata in “bytes transferred” so INT8/INT4 are not under‑counted.
- **Milestone 3 simplification**: use `proportion/order_mode` in `kv_cache_index` for cheap cache‑length ablations before adding top‑k pruning.
- **Budget curves**: report accuracy vs *effective* bytes (precision × cache length + metadata).
- **QAT scope**: keep projector‑only training (freeze base/teacher weights) to stay within 1×H100 budget.
- **Mixed precision**: start with “last‑N layers FP16, rest INT8” as the minimal schedule.
- **Heterogeneity**: plan for tokenizer alignment (`is_do_alignment`) when crossing model families.
- **Local vs GPU accuracy**: MPS local runs can show degenerate answer distributions (e.g., over‑predicting “A”). Treat local runs as pipeline validation only; rely on GPU evals for accuracy.
- **include_response alignment**: paper eval defaults to `include_response=false`. Our code now supports both; keep official evals at false, use true only for ablations.
- **Prompt alignment**: stay on the unified Non‑CoT prompt template with `use_template=true`, `use_cot=false`, `max_new_tokens=64`, temp=0.
- **Cache/conda drift**: HF caches are large; keep caches out of git and ensure conda env matches versions in `env_info.json`.
- **Extensions**: add `--eval-limit` flag to the runner for GPU smoke runs and capture answer‑distribution stats (A/B/C/D counts) automatically.

## Time Budget (rough)
- Workshop path: ~1 day on 1 H100 (assuming caches are warm).
- Main‑conf path: ~3–5 days on 1 H100.
