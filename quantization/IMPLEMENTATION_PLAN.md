# Implementation Plan + Experiment Matrix (1x H100 per milestone)

Goal: deliver a workshop‑ready paper on **Quantized Cache‑to‑Cache** with communication‑budget curves, and a clear path to a main‑conference submission (QAT + mixed precision + heterogeneity scaling). Each milestone is designed to run on a single H100.

## Status (2026‑01‑28)
- **M0–M4 complete** on OpenBookQA + ARC‑C with full runs (see `quantization/golden/golden_summary.md`).
- **M5 QAT**: training smoke only (local); **GPU QAT eval pending**.
- **M6 mixed precision**: **needs re‑run** with corrected 28‑layer schedule (last‑4 = layers 24–27).
- **M7 heterogeneity**: **not complete** — current results are an alignment ablation on the same model pair (accuracy drop); needs true cross‑family pair (e.g., Llama/Gemma‑3‑1B‑IT), plus alignment on/off.
- **M8 selective transfer**: **partial** — p=1.0 complete (both datasets), p=0.5 complete (both datasets), remaining proportions not run.
- **Registry**: `quantization/registry/run_registry.json` updated for completed full runs; incomplete datasets intentionally left out so they will rerun.

## Milestone Status Details (run tags)
- **M0 baseline**: `step0_20260127_214552` complete (OpenBookQA + ARC‑C).
- **M2 PTQ**: `step1_int8_20260127_214552`, `step1_int4_20260127_214552` complete.
- **M3 pruning grid**: complete across front/back at p={1.0,0.75,0.5,0.25,0.10}; p0.10 front/back from `20260127_082712/084143`, others from `20260127_214552`.
- **M4 curves**: `quantization/analysis/m4_budget_curve/budget_curve_{openbookqa,arc_c}.png` generated.
- **M6 mixed precision**: `step6_int8_20260128_011432` complete (INT8 + last‑4 FP16 layers) **but schedule file updated; re‑run recommended**.
- **M7 alignment ablation**: `step7_20260128_011432` complete (same‑pair alignment; not true heterogeneity).
- **M8 selective transfer**: `step8_int8_proj_vnorm_topk_p1p0_20260128_011432` complete; `p0p5` OpenBookQA + ARC‑C complete.

## Data Capture Contract (applies to every milestone)
- Each run creates `quantization/data/step_X_<name>/<run_tag>/` with (folder names remain `step_X` for now to match scripts):
  - `configs/` (the exact eval/train configs used)
  - `logs/` (stdout/stderr logs)
  - `results/` (JSON outputs)
  - `manifests/` (checkpoint + environment provenance)
  - `manifests/system_info.json` (GPU + driver snapshot)
  - `manifests/timings.json` (per‑dataset wall‑clock timings)
  - `manifests/*_manifest.json` includes `bytes_estimate` (bytes/token + effective proportion; multiply by avg length for per‑sequence bytes)
- Commit `quantization/data/step_*` to git after each milestone so results are reviewable.
- Cleanup hygiene: remove failed runs that do not contain `results/` or have `status=failed` in manifests (keep only validated runs).
- Registry keys now include **M6 schedule hash**, **M7 alignment flag**, and **M8 scope/sparse‑fuse flags** so ablations don’t overwrite each other.

## Milestone Execution Checklist (clean + verified)
Use this as the runbook. Replace `$PROJECT_ROOT` and cache paths as needed (RunPod `/workspace/LatentWire`, HPC `/projects/.../LatentWire`).

### Common prep (once per machine)
- [ ] `git clone git@github.com:SujeethJinesh/LatentWire.git && cd LatentWire`
- [ ] `git submodule update --init --recursive quantization/C2C`
- [ ] (Recommended) caches:
  - `export HF_HOME=/scratch/$USER/.cache/huggingface`
  - `export C2C_CKPT_ROOT=/scratch/$USER/c2c_checkpoints`

### GPU verification pass (smoke, all milestones)
This is the “make sure nothing breaks” pass before full runs. **Now unified in SLURM.**
- [ ] **All milestones via SLURM (smoke):**
  - `RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 RUN_MILESTONE_5=1 RUN_MILESTONE_6=1 RUN_MILESTONE_7=1 RUN_MILESTONE_8=1 EVAL_SMOKE=1 SMOKE_LIMIT=50 sbatch quantization/submit_milestones.slurm`
  - Optional overrides:  
    - `M5_RECIPE=quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8_smoke.json`  
    - `M6_LAYER_SCHEDULES="quantization/configs/kv_layer_schedule.yaml quantization/configs/kv_layer_schedule_last2.yaml quantization/configs/kv_layer_schedule_last8.yaml"`  
    - `M7_BASE_MODEL=Qwen/Qwen3-0.6B M7_TEACHER_MODEL=meta-llama/Llama-3.2-1B-Instruct M7_ALIGNMENT_MODES="0 1"`  
    - `M8_SELECT_MODES="front vnorm_topk proj_vnorm_topk" M8_SELECT_PROPORTIONS="1.0 0.5 0.25" M8_SELECT_SCOPES="prompt" M8_SPARSE_FUSE_MODES="1"`

### Full runs (paper‑quality)
- [ ] **All milestones via SLURM (full, skip done by default):**
  - `RUN_FULL=1 RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 RUN_MILESTONE_5=1 RUN_MILESTONE_6=1 RUN_MILESTONE_7=1 RUN_MILESTONE_8=1 sbatch quantization/submit_milestones.slurm`
  - Optional: `SKIP_DONE=0` to ignore registry, `FORCE_RERUN=1` to override skip.
- [ ] **Auto‑push results after each milestone (full runs only):**
  - `AUTO_PUSH=1` (default: off)
  - Optional: `AUTO_PUSH_REBASE=1` (default), `AUTO_PUSH_FULL_ONLY=1` (default), `AUTO_PUSH_REMOTE=origin`, `AUTO_PUSH_BRANCH=main`
- [ ] **Registry file (auto‑updated):**
  - `quantization/registry/run_registry.json` (per‑dataset status with run root + config hash)
- [ ] **M4 analysis:**
  - `python quantization/scripts/analyze_budget_curve.py --runs-root quantization/data --output-dir quantization/analysis/m4_budget_curve`
- [ ] **M5 full QAT:**
  - `python quantization/C2C/script/train/SFT_train.py --config quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8.json`
  - then `python quantization/scripts/run_step1_kv_ptq.py --kv-quant-scheme int8`
- [ ] **M6 full mixed precision:**
  - `python quantization/scripts/run_step1_kv_ptq.py --kv-quant-scheme int8 --kv-quant-layer-schedule quantization/configs/kv_layer_schedule.yaml`
- [ ] **M7 full heterogeneity:**
  - `python quantization/scripts/run_step1_kv_ptq.py --base-model Qwen/Qwen3-0.6B --teacher-model meta-llama/Llama-3.2-1B-Instruct --kv-quant-scheme int8`

### Post‑run hygiene
- [ ] Archive “golden” results: `mkdir -p quantization/golden && cp -r quantization/data/step_*_*/<run_tag> quantization/golden/`
- [ ] Clean failed runs: `python quantization/scripts/run_step1_kv_ptq.py --cleanup-failed`

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
- Login node prep (verbose streaming):  
  - `conda run --no-capture-output -n rosetta python -u quantization/scripts/run_step0_baselines.py --prep-only`  
- GPU node eval: `python quantization/scripts/run_step0_baselines.py`

**GPU node instructions (Milestone 0, direct run)**  
1) Allocate a GPU node (example):  
   - `salloc -N 1 -G 1 -A marlowe-m000066 -p preempt --time=3:00:00 --mem=32GB`  
2) From the GPU node:  
   - `cd /projects/m000066/sujinesh/LatentWire`  
   - Optional cache overrides:  
     - `export HF_HOME=/scratch/m000066/$USER/.cache/huggingface`  
     - `export C2C_CKPT_ROOT=/scratch/m000066/$USER/c2c_checkpoints`  
   - Run eval: `python quantization/scripts/run_step0_baselines.py`  
3) Outputs land in `quantization/data/step_0_baselines/<run_tag>/` with logs, configs, results, and manifests.

**GPU node instructions (Milestone 0, SLURM batch)**  
- `RUN_MILESTONE_0=1 sbatch quantization/submit_milestones.slurm`  
- Use `PREFLIGHT=1` (default) to validate plumbing first.  
- For a preflight‑only check: `RUN_MILESTONE_0=1 DRY_RUN=1 sbatch quantization/submit_milestones.slurm`

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

**What it shows**  
Whether C2C is robust to shorter transmitted caches and how accuracy degrades as we remove early vs late instruction tokens.

**Expected outcome**  
- Accuracy should drop as `kv_cache_proportion` decreases; if it does not, we may be over‑pruning already or the task is insensitive.  
- `order_mode=front` vs `back` should reveal whether early or late instruction content is more valuable for this model pair.  

**Testing approach**  
Phases 2–6 below provide the local sanity checks and GPU smoke/full runs; local runs validate plumbing, GPU runs supply paper‑quality accuracy.

**Milestone 3 sub‑steps (phased)**  
- **Phase 0 (Plan update)**: lock the cache‑length grid and defaults.  
  - **Default**: `kv_cache_proportion=1.0`, `kv_cache_order_mode=front`.  
  - **Grid**: {1.0, 0.75, 0.5, 0.25, 0.1} × {front, back}.  
  - **Why**: aligns with the “budget curve” goal while keeping the search small enough for 1×H100.  
  - **Extensions (backburner)**: add `order_mode=middle` (center chunk) or random‑keep at one proportion (e.g., 0.5) for interpretability.  
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
Compute bytes per sequence from model config + quant scheme + cache proportion, then plot accuracy vs bytes for each dataset.

**What it shows**  
The concrete trade‑off between bandwidth cost and accuracy, enabling direct comparison across INT8/INT4 and cache‑length pruning.

**Expected outcome**  
- Accuracy should degrade monotonically as bytes decrease (small deviations are noise).  
- INT8 with shorter caches should outperform INT4 at the same byte budget unless quantization error dominates.  

**Milestone 4 sub‑steps (phased)**  
- **Phase 0 (Budget definition)**: define byte formula and the metadata required.  
  - **Formula**: `bytes = avg_input_length × kv_cache_proportion × 2 × num_layers × num_kv_heads × head_dim × bytes_per_element`.  
  - **Why**: fixes the accounting so curves are comparable and reproducible.  
- **Phase 1 (Analysis script)**: add a parser to read run folders, extract accuracy + length stats, compute bytes, and emit CSV + plots.  
  - **Why**: makes the curve generation repeatable from raw results.  
- **Phase 2 (Local validation)**: run the script in demo mode to validate plotting and CSV formatting without GPU results.  
  - **Why**: ensures the analysis pipeline works before the long GPU runs finish.  
- **Phase 3 (GPU analysis)**: run the script on M2 + M3 GPU outputs, generating final plots for the paper.  
  - **Why**: produces the main empirical figure for the workshop paper.

**Telemetry (Milestone 4)**  
- `budget_curve.csv`: per‑run rows with accuracy, bytes, and config metadata.  
- `budget_curve_<dataset>.png`: accuracy vs bytes plots per dataset.  

**Local test commands (Milestone 4)**  
- Demo mode (no GPU results needed):  
  - `python quantization/scripts/analyze_budget_curve.py --demo`

**GPU analysis commands (Milestone 4, deferred)**  
- After GPU runs are complete:  
  - `python quantization/scripts/analyze_budget_curve.py --runs-root quantization/data`

**GPU node instructions (Milestone 4 analysis)**  
1) From any node (GPU not required, but fine if already allocated):  
   - `cd /projects/m000066/sujinesh/LatentWire`  
   - Optional: set offline mode if the node has no internet:  
     - `export TRANSFORMERS_OFFLINE=1`  
2) Run analysis:  
   - `python quantization/scripts/analyze_budget_curve.py --runs-root quantization/data --output-dir quantization/analysis/m4_budget_curve`  
3) Outputs:  
   - `quantization/analysis/m4_budget_curve/budget_curve.csv`  
   - `quantization/analysis/m4_budget_curve/budget_curve_<dataset>.png`  

**Status (Milestone 4)**  
- **Complete**: budget curves generated at `quantization/analysis/m4_budget_curve/` (`budget_curve.csv`, `budget_curve_openbookqa.png`, `budget_curve_arc_c.png`).  

## Milestone 5: QAT Recovery (Main‑conf extension)
**What**  
Quantization‑aware training of the projector under INT8 noise.

**Why**  
Shows that low‑precision transfer can be learned, not just approximated.

**How**  
Train on a small subset (10–50k samples), then re‑evaluate ARC‑C + OpenBookQA.

**Configs (added)**  
- Full QAT config: `quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8.json`  
- Local smoke config (Mac): `quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8_smoke.json`  

**Local smoke test (Mac/MPS)**  
Purpose: validate the QAT wiring (kv_quant_config flows into RosettaModel) without full training.  
Command (from repo root):  
- `python quantization/C2C/script/train/SFT_train.py --config quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8_smoke.json`  
Notes: smoke config uses `dtype: fp32` and `device: cpu` to avoid MPS reduction bugs on Mac.

**GPU training (RunPod/H100)**  
Command (from repo root):  
- `python quantization/C2C/script/train/SFT_train.py --config quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8.json`  

**Workshop+ (optional) pre‑QAT main‑conf previews**  
If we want to strengthen the workshop paper without committing to full QAT, we can do one or more of the following *before* Milestone 5 training. These reuse the existing PTQ runner and add minimal code paths.

- **M5‑P0: NF4 PTQ ablation (low risk)**  
  - **Why**: NF4 often recovers INT4 accuracy with similar bandwidth.  
  - **What**: add `scheme: nf4` to `kv_quant_config` (likely via bitsandbytes).  
  - **How**: implement NF4 quant/dequant in `rosetta/utils/quant.py`, run OpenBookQA + ARC‑C like INT4.  
  - **Success criterion**: NF4 accuracy ≥ INT4, bandwidth unchanged.

- **M5‑P1: Lightweight mixed‑precision schedule (medium risk)**  
  - **Why**: later layers are more sensitive; mixed precision can lift accuracy per byte.  
  - **What**: add per‑layer precision schedule (e.g., last 4 layers FP16, others INT8/INT4).  
  - **How**: extend `kv_quant_config` with a `layer_schedule` map; apply in `wrapper.py`.  
  - **Success criterion**: measurable accuracy gain at similar or modestly higher bytes.

- **M5‑P2: Quantize both source + base KV (medium risk)**  
  - **Why**: tests worst‑case compression and isolates where quantization noise matters.  
  - **What**: add a `quantize_target: source|base|both` knob.  
  - **How**: apply quantization on both cache paths in `wrapper.py`; re‑run INT8/INT4.  
  - **Success criterion**: accuracy drop is bounded; provides stronger budget curves.

**Main‑conf focus (recommended priority)**  
- **M5‑Core: QAT projector under INT8** (primary algorithmic contribution).  
- **M6‑Core: Mixed‑precision schedule** (accuracy‑per‑byte improvement).  
- **Systems add‑on**: add **measured bandwidth/latency** (H100) and/or **true bit‑packing** to strengthen the systems contribution.  
- **M7‑Core: Heterogeneity scaling** if we have bandwidth; otherwise defer to appendix.

**Additional Main‑conf extensions (beyond M5–M7)**  
These are *lower‑priority* or appendix‑level once M5/M6 are complete:  
- **NF4**: accuracy‑preserving INT4 variant (likely via bitsandbytes) to strengthen INT4 results with low added risk.  
- **FP8 (H100‑friendly)**: hardware‑optimized precision that can narrow the INT8/INT4 accuracy gap; higher engineering effort.  
- **Quantize both source+base KV**: more aggressive compression; likely larger drop but stronger budget curves.  
- **True bit‑packing**: makes byte accounting realistic (not just fake‑quant); higher engineering overhead.  

## Milestone 6: Mixed Precision by Layer (Main‑conf extension)
**What**  
Use higher precision in later layers and lower precision in early layers.

**Why**  
Leverages layer sensitivity to improve the accuracy‑per‑byte curve.

**How**  
Add a per‑layer precision schedule to `kv_quant_config` and evaluate a small, targeted grid.  
Start with INT8 as the default and promote only the last‑N layers to FP16.

**Proposed config shape**  
```yaml
kv_quant_config:
  enabled: true
  scheme: int8       # default scheme
  axis: head
  layer_schedule:
    default: int8
    overrides:
      - layers: [24,25,26,27]
        scheme: fp16
```
**Tier‑1 / Tier‑2 plan**  
- **Tier‑1 (must‑run):** last‑4 FP16 over INT8 baseline using `quantization/configs/kv_layer_schedule.yaml`.  
- **Tier‑2 (nice‑to‑have):** last‑2 and last‑8 FP16 using:  
  - `quantization/configs/kv_layer_schedule_last2.yaml`  
  - `quantization/configs/kv_layer_schedule_last8.yaml`  
**Why this grid**: last‑4 is a common sensitivity band; last‑2 tests minimal overhead; last‑8 probes diminishing returns.

**Milestone 6 phases (recommended)**  
- **M6‑P0 (Design)**: pick schedule grid and metrics.  
  - Grid: last‑4 FP16 (Tier‑1), last‑2 + last‑8 FP16 (Tier‑2).  
  - Why: small, interpretable grid; likely accuracy gain with modest byte increase.  
- **M6‑P1 (Implementation)**:  
  - Parse `layer_schedule` in `rosetta/utils/quant.py` and apply per‑layer scheme.  
  - Update `wrapper.py` and `oracle.py` to choose scheme by **target layer index** (so we can express “last‑N layers higher precision”).  
  - Update `run_step1_kv_ptq.py` to accept `--kv-quant-layer-schedule` (YAML/JSON) and `--kv-quant-last-fp16 N`, and to inject the schedule into `kv_quant_config`.  
  - Update `analyze_budget_curve.py` to compute **effective bits per element** when mixed precision is used (average bits across layers).  
  - Log the resolved schedule into run manifests.  
- **M6‑P2 (Local sanity)**:  
  - Run **1–5 samples** on Mac (`--mode local`) to validate schedule wiring.  
  - Compare outputs vs INT8 baseline to ensure no crashes.  
- **M6‑P3 (GPU smoke)**:  
  - Run 25–50 samples on OpenBookQA with last‑4 FP16 to check throughput + correctness.  
- **M6‑P4 (Full GPU grid)**:  
  - OpenBookQA + ARC‑C for last‑4 FP16 (Tier‑1), then last‑2/last‑8 FP16 (Tier‑2).  
  - Compare accuracy vs INT8 and compute bytes‑per‑sequence deltas (for M4 curves).  

**Expected outcome**  
- Last‑4 FP16 should recover some accuracy with a modest byte increase.  
- Last‑8 FP16 may improve further but with diminishing returns.  

**Testing commands (local)**  
- INT8 baseline (1 sample):  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-num-samples 1 --kv-quant-scheme int8`  
- Mixed precision (last‑4 FP16):  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-num-samples 1 --kv-quant-scheme int8 --kv-quant-last-fp16 4`  
- Custom schedule (optional):  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-num-samples 1 --kv-quant-layer-schedule quantization/configs/kv_layer_schedule.yaml`

## Milestone 7: Heterogeneity Scaling (Main‑conf extension)
**What**  
Test at least one cross‑family pair (e.g., Qwen3 ← Llama3.2).

**Why**  
Demonstrates that the method generalizes beyond a single model family.

**How**  
Reuse the PTQ + pruning settings from Steps 2–3, but swap the **teacher model family** and enable tokenizer alignment when needed.

**Milestone 7 phases (recommended)**  
- **M7‑P0 (Design)**: choose 1–2 cross‑family pairs + datasets.  
  - Example pairs: Qwen3‑0.6B ← Llama‑3.2‑1B, Qwen3‑0.6B ← Gemma‑3‑1B‑IT.  
  - Why: small, interpretable set; enough to show generalization.  
- **M7‑P1 (Config)**: add a dedicated eval recipe (copy of `unified_eval.yaml`) with:  
  - `rosetta_config.teacher_model` set to the new source model.  
  - `is_do_alignment=true` if tokenizers differ (cross‑family).  
  - Keep `use_template=true`, `use_cot=false`, `max_new_tokens=64`, `temperature=0`.  
- **M7‑P2 (Local sanity)**: run **1–5 samples** in local mode with the new base/teacher to verify:  
  - tokenizer alignment, prompt formatting, and that outputs are non‑degenerate.  
- **M7‑P3 (GPU smoke)**: run 25–50 samples on OpenBookQA to validate throughput and correctness.  
- **M7‑P4 (Full GPU eval)**: run OpenBookQA + ARC‑C (INT8) for the cross‑family pair.  
- **M7‑P5 (Analysis)**: compare to within‑family INT8 baseline; report delta accuracy + bytes.  
**Tier‑1 / Tier‑2 plan**  
- **Tier‑1 (must‑run):** same pair (Qwen3‑0.6B ← Llama‑3.2‑1B) with **alignment on/off** to show alignment sensitivity.  
- **Tier‑2 (nice‑to‑have):** add Gemma‑3‑1B‑IT (with alignment) for broader heterogeneity.

**Expected outcome**  
- Cross‑family should show **some drop** vs within‑family but remain usable.  
- Alignment flag should reduce severe degradation; if not, consider alternative token alignment or prompt templates.

**Local test commands (M7)**  
- `python quantization/scripts/run_step1_kv_ptq.py --mode local --local-dataset openbookqa --local-num-samples 1 --base-model Qwen/Qwen3-0.6B --teacher-model meta-llama/Llama-3.2-1B-Instruct --kv-quant-scheme int8`  
- Repeat with `--teacher-model google/gemma-3-1b-it` if desired.

**GPU commands (M7, deferred)**  
- Create a new eval recipe (copy `quantization/C2C/recipe/eval_recipe/unified_eval.yaml` → `unified_eval_hetero.yaml`) with the new teacher model and `is_do_alignment=true`.  
- Then run via the standard runner (INT8):  
  - `python quantization/scripts/run_step1_kv_ptq.py --mode gpu --kv-quant-scheme int8`  
  - (If multiple recipes are needed, run each by swapping the eval recipe or adding a flag later.)

## Milestone 8: Selective & Compressed Cache Transfer (Token-Importance Sparse C2C)
**What**  
Add token-level selective transfer of the sharer KV cache into C2C, combined with KV PTQ (INT8/INT4). Unlike Milestone 3 (front/back truncation), this selects a **sparse subset of token positions** to transmit/fuse and optionally performs **sparse fusion** (fuser runs only on selected tokens, then scatter back).

**Why**  
Milestones 1–4 give "precision x length" budget curves, but length pruning is structural and may drop high-value tokens. Token-importance selection should preserve accuracy at the same bytes. Sparse fusion reduces both **communication bandwidth** and **prefill compute**.

**Novelty (vs KVComm/Q-KVComm)**  
KVComm focuses on layer selection within same-architecture sharing. Here we keep C2C's **projection + fusion** for **heterogeneous** models and add a **token-level** sparsity axis inside the C2C fuser.

**Config block (new)**  
```yaml
kv_transfer_config:
  enabled: true
  token_select_mode: vnorm_topk   # vnorm_topk | knorm_topk | proj_vnorm_topk | random | front | back
  token_select_proportion: 0.25
  token_select_scope: prompt      # prompt | instruction_only | all_context
  token_select_min_tokens: 64
  sparse_fuse: true               # fuse only selected tokens, then scatter back
  scatter_fill: receiver_only     # receiver_only | zeros
  kv_quant_config:
    enabled: true
    scheme: int8                  # int8 | int4
    axis: head
    eps: 1e-6
  index_dtype_bytes: 2            # uint16 indices if seq_len < 65535
  include_scale_overhead: true
```

**Milestone 8 phases (recommended)**  
- **M8‑P0 (Design lock)**: modes = {front, back, random, vnorm_topk, knorm_topk, proj_vnorm_topk}, proportions = {1.0, 0.5, 0.25, 0.10}, schemes = {int8, int4}.  
- **M8‑P1 (Token selection)**: add `rosetta/utils/kv_select.py` with scoring + top‑k selection and scope masking; add projector‑aware scoring via `proj_vnorm_topk` (score in receiver space after projection).  
- **M8‑P2 (Sparse transfer + fuse)**: gather selected KV, quantize gathered KV, project + fuse selected tokens only, scatter back.  
- **M8‑P3 (Runner + telemetry)**: add `run_step8_selective_transfer.py` with token stats + effective bytes.  
- **M8‑P4 (Local sanity)**: 1–5 samples; `proportion=1.0` must match Milestone 1 outputs.  
- **M8‑P5 (GPU smoke)**: 50‑sample runs, INT8 x {1.0, 0.5, 0.25}, compare {front, random, vnorm_topk, knorm_topk, proj_vnorm_topk}.  
- **M8‑P6 (Full GPU grid + plots)**:  
  - INT8 x {1.0, 0.5, 0.25, 0.10} x {front, vnorm_topk, proj_vnorm_topk}.  
  - INT8 x {0.25, 0.10} x {random, knorm_topk}.  
  - INT4 x {1.0, 0.5, 0.25} x {front, vnorm_topk}.  
  - Extend `analyze_budget_curve.py` with token_select columns and effective bytes (payload + index bytes + quant scales).

**Tier‑1 / Tier‑2 plan**  
- **Tier‑1 (must‑run):** INT8 × {1.0, 0.5, 0.25} × {front, vnorm_topk, proj_vnorm_topk} on OpenBookQA + ARC‑C.  
- **Tier‑2 (nice‑to‑have):** INT8 × {0.25, 0.10} × {random, knorm_topk} and INT4 × {1.0, 0.5, 0.25} × {front, vnorm_topk}.  
**Why**: Tier‑1 gives the core projector‑aware vs structural baseline; Tier‑2 adds robustness and the INT4 compression axis.

**Why these ablations**  
- **front**: ties directly to M3 (length pruning) and serves as a structured baseline.  
- **random**: sanity check to show selection matters vs arbitrary sparsity.  
- **knorm_topk**: tests whether K‑norm is a viable proxy vs V‑norm (robustness of the selection metric).  
- **proj_vnorm_topk**: C2C‑native, projector‑aware scoring to claim novelty over single‑model KV selection.

**Expected outcome**  
- At matched bytes, projector‑aware `proj_vnorm_topk` should match or exceed `vnorm_topk` and outperform `front/back`.  
- `random` provides a sanity baseline; it should underperform importance‑based scores at the same proportion.  
- `knorm_topk` tests sensitivity to K‑norm vs V‑norm; expect similar but slightly worse than V‑norm (V tends to correlate with utility in prior KV work).
- Sparse fusion should reduce fuser compute roughly proportional to `token_select_proportion`.
**Notes / risks**  
- `proj_vnorm_topk` computes a full projection to score tokens, which adds overhead. If it becomes too expensive, add a coarse scoring cadence (e.g., score every N tokens) or a cap on selected tokens beyond `token_select_min_tokens`.
**Telemetry (M8)**  
- Record `token_select_mode`, `token_select_proportion`, `token_select_scope`, `sparse_fuse`, `scatter_fill`, and selection stats (mean/min/max tokens) in run manifests and result summaries.  
- Use these fields in `analyze_budget_curve.py` to compute **effective bytes** (cache proportion × token select proportion × quant bits + index/scale overhead).

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
| E2 | Qwen3‑0.6B ← Gemma‑3‑1B‑IT | ARC‑C | C2C + PTQ | INT8 | full | no |

---

## Workshop vs Main‑Conference Path
- **Workshop**: Milestones 0 → 1 → 2 → 3 → 4 (baseline + PTQ + pruning + budget curve).
- **Main‑conf**: Workshop path + M5 (QAT) + M6 (mixed precision) + **systems measurements** (bandwidth/latency or bit‑packing).  
  - **M7 (heterogeneity)** is recommended if time allows; otherwise move to appendix.  
  - **M8 (token‑level sparse C2C)** is a strong main‑conf differentiator if bandwidth permits.

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

---

## Performance Speedups (scoped plan + Mac correctness tests)
Goal: increase GPU utilization safely (batching/compile/prefetch), while proving correctness on Mac before GPU runs.

### Step 0 — Feasibility research (local)
- Inspect `quantization/C2C/script/evaluation/unified_evaluator.py` for:
  - whether it already supports batched decoding
  - where prompts are built vs generated
  - where logits/outputs are post‑processed
- Decide on **batching strategy**:
  1) **Receiver‑only batching** first (lowest risk).
  2) Extend to **C2C batched decoding** if outputs match.

### Step 1 — Correctness harness (Mac)
- Add a small local test script to compare **batched vs unbatched outputs** on ~20 samples:
  - Dataset: OpenBookQA
  - Config: `use_cot=false`, `use_template=true`, `max_new_tokens=64`, temp=0
  - Compare:
    - exact output strings
    - extracted answers
    - accuracy (should match exactly)
- If mismatch: inspect for padding/attention mask or stop‑token issues; fix before GPU use.

### Step 2 — Implement batching (guarded)
- Add `--eval-batch-size` to unified evaluator and wire it into config.
- Default remains **1** (no behavior change).
- Only enable batching after Mac correctness passes.

### Step 3 — Optional compile speedup
- Add an opt‑in flag `--torch-compile` for the projector/fuser path.
- Validate numerically on Mac or small GPU sample (20 items).

### Step 4 — Deploy on GPU
- Use `eval.batch_size=4` or `8`, then scale up while monitoring VRAM.
- Keep `EVAL_SMOKE=1` for first run; compare to unbatched accuracy.

### Expected gain (rough)
- Batching: **2×–4×** throughput
- Compile: **+10–40%**
- Combined: **~2×–5×** end‑to‑end speedup

## References (ArXiv)

### Quantization
- GPTQ: Accurate Post‑Training Quantization for Generative Pre‑trained Transformers — https://arxiv.org/abs/2210.17323
- SmoothQuant: Accurate and Efficient Post‑Training Quantization for Large Language Models — https://arxiv.org/abs/2211.10438
- AWQ: Activation‑aware Weight Quantization for LLMs — https://arxiv.org/abs/2306.00978
- LLM.int8(): 8‑bit Matrix Multiplication for Transformers at Scale — https://arxiv.org/abs/2208.07339
- ZeroQuant: Efficient and Affordable Post‑Training Quantization for Large‑Scale Transformers — https://arxiv.org/abs/2206.01861

### Cross‑LLM Communication / Multi‑Agent
- AutoGen: Enabling Next‑Gen LLM Applications via Multi‑Agent Conversation Framework — https://arxiv.org/abs/2308.08155
- CAMEL: Communicative Agents for “Mind” Exploration of LLM Society — https://arxiv.org/abs/2303.17760
- MetaGPT: Meta Programming for A Multi‑Agent Collaborative Framework — https://arxiv.org/abs/2308.00352
- FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance — https://arxiv.org/abs/2305.05176
- HuggingGPT: Solving AI Tasks with ChatGPT and Its Friends — https://arxiv.org/abs/2303.17580

### KV‑Cache / Long‑Context Memory
- Cache‑to‑Cache (C2C): Direct Semantic Communication via KV‑Cache Fusion — https://arxiv.org/abs/2510.03215
- vLLM / PagedAttention: Efficient Memory Management for LLM Serving — https://arxiv.org/abs/2309.06180
- StreamingLLM: Efficient Streaming LMs with Attention Sinks — https://arxiv.org/abs/2309.17453
- Transformer‑XL: Attentive LMs Beyond a Fixed‑Length Context — https://arxiv.org/abs/1901.02860
- Compressive Transformer: Long‑Range Sequence Modeling — https://arxiv.org/abs/1911.05507
- Memorizing Transformers — https://arxiv.org/abs/2203.08913

### KV‑Cache Communication / Compression
- KVComm: Selective KV Sharing for Efficient LLM Communication — https://arxiv.org/abs/2510.03346
- Q‑KVComm: Adaptive KV Cache Compression for Multi‑Agent Communication — https://arxiv.org/abs/2512.17914
- Latent Space Communication via KV Cache Alignment — https://arxiv.org/abs/2601.06123
- ZipCache: Efficient KV Cache Compression via Token Saliency — https://arxiv.org/abs/2405.14256
- TokenSelect: Efficient Long‑Context Inference via Token‑Level KV Selection — https://aclanthology.org/2025.emnlp-main.1079/
- VATP: Attention Score is not All You Need for Token Importance (Value Matters) — https://arxiv.org/abs/2406.12335
- PDTrim: Prefill‑Decode KV Pruning for Communication Bandwidth — https://arxiv.org/abs/2509.04467

### Inference / Decoding / Serving
- FlashAttention: Fast and Memory‑Efficient Exact Attention — https://arxiv.org/abs/2205.14135
- FlashAttention‑2: Faster Attention with Better Parallelism — https://arxiv.org/abs/2307.08691
- Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads — https://arxiv.org/abs/2401.10774
- DeepSpeed Inference: Efficient Inference of Transformer Models at Scale — https://arxiv.org/abs/2207.00032
- Speculative Sampling / Decoding — https://arxiv.org/abs/2302.01318
