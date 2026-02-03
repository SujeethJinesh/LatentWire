# Implementation Plan + Experiment Matrix (1x H100 per milestone)

Goal: deliver a workshop‑ready paper on **Quantized Cache‑to‑Cache** with communication‑budget curves, and a clear path to a main‑conference submission (QAT + mixed precision + heterogeneity scaling). Each milestone is designed to run on a single H100.

## Status (2026‑02‑01)
- **M0–M4 complete** on OpenBookQA + ARC‑C with full runs (see `quantization/golden/golden_summary.md`).
- **M5 QAT**: full eval still weak (projector‑only INT8); needs recipe tuning or longer training.
- **M6 mixed precision**: complete across last‑2/last‑4/last‑8 schedules (near baseline).
- **M7 heterogeneity**: partial — alignment on/off done for same‑pair; hetero pair (Qwen3$\rightarrow$Llama3.2) align‑on reported, align‑off unstable; needs a clean hetero run.
- **M8 selective transfer**: full grid complete (p={1.0,0.5,0.25,0.10} for front/random/knorm/vnorm/proj\_vnorm; INT8 + INT4).
- **M9 delta selection**: full grid complete + p=0.05 (OpenBookQA only). Hetero spot check pending.
- **M10 RD‑C2C**: budgets complete at {1/32, 1/16, 1/8, 1/4}. Hetero spot check pending.
- **Registry**: `quantization/registry/run_registry.json` updated for completed full runs; incomplete datasets intentionally left out so they will rerun.

## Live Action Items (rolling)
- **Hetero M9/M10 spot checks** (Qwen3‑0.6B ← Llama‑3.2‑1B‑Instruct; at least p=0.25 for M9 and 1/8 for M10).
- **System metrics**: capture prefill/selection/projection/fuse/end‑to‑end timings and report in paper.
- **Stability checks**: 2–3 seeds (or shard bootstrap) on 1–2 key points (M9 p=0.10, M10 1/16).
- **Extra dataset**: add one more benchmark (gsm8k is supported end‑to‑end in the evaluator).
- **RD ablation**: compare {drop,int8} vs {drop,int4,int8} at the same budgets.
- **Run true M7 heterogeneity** (clean align on/off on a cross‑family pair).
- **Update registry + golden summary** after each completed block.

## Main‑Conference Completeness Checklist
- **DONE**: M0–M4 baselines + budget curves (OpenBookQA, ARC‑C).
- **DONE**: M6 mixed‑precision schedules (last‑2/last‑4/last‑8).
- **DONE**: M8 selective‑transfer grids (INT8/INT4; p={1.0,0.5,0.25,0.10}).
- **DONE**: M9 delta selection grid + p=0.05 on OpenBookQA.
- **DONE**: M10 RD budgets {1/32,1/16,1/8,1/4} on OpenBookQA + ARC‑C.
- **PENDING**: M7 hetero align on/off with correct fuser checkpoint (clean run).
- **PENDING**: M9 hetero spot check (p=0.25 or p=0.10).
- **PENDING**: M10 hetero spot check (budget 1/8).
- **PENDING**: RD ablation {drop,int8} vs {drop,int4,int8}.
- **PENDING**: System metrics (timing sync + table in paper).
- **PENDING**: Stability checks (two shards/seeded subsets).
- **PENDING**: Extra dataset breadth (gsm8k).
- **PENDING**: QAT recovery (improved recipe or longer training).

## Milestone Status Details (run tags)
- **M0 baseline**: `step0_20260127_214552` complete (OpenBookQA + ARC‑C).
- **M2 PTQ**: `step1_int8_20260127_214552`, `step1_int4_20260127_214552` complete.
- **M3 pruning grid**: complete across front/back at p={1.0,0.75,0.5,0.25,0.10}; p0.10 front/back from `20260127_082712/084143`, others from `20260127_214552`.
- **M4 curves**: `quantization/analysis/m4_budget_curve/budget_curve_{openbookqa,arc_c}.png` generated.
- **M6 mixed precision**: complete across last‑2/last‑4/last‑8 schedules (see `golden_summary.md`).
- **M7 alignment ablation**: `step7_20260128_011432` complete (same‑pair alignment; not true heterogeneity).
- **M8 selective transfer**: full grid complete (see `golden_summary.md`).
- **M9 delta selection**: `step9_*_20260131_202705_m9m10` complete (full grid + p=0.05 OpenBookQA).
- **M10 RD‑C2C**: `step10_*_20260131_202705_m9m10` complete (budgets 1/32–1/4).

## Data Capture Contract (applies to every milestone)
- Each run creates `quantization/data/step_X_<name>/<run_tag>/` with (folder names remain `step_X` for now to match scripts):
  - `configs/` (the exact eval/train configs used)
  - `logs/` (stdout/stderr logs)
  - `results/` (JSON outputs)
  - `manifests/` (checkpoint + environment provenance)
  - `manifests/system_info.json` (GPU + driver snapshot)
  - `manifests/timings.json` (per‑dataset wall‑clock timings)
  - `manifests/*_manifest.json` includes byte + timing accounting (required for new runs):
    - `bytes_estimate` (legacy estimate only; do **not** use for head-to-head claims)
    - `bytes_estimated_total` (preferred estimated field; same semantics as `bytes_estimate`)
    - `bytes_measured_total` (canonical; exact bytes on wire: `len(blob)` for KVWire or UTF-8 bytes for text-message baselines)
    - `bytes_measured_breakdown` (optional dict: indices/payload/scales/headers)
    - `wire_encode_ms`, `wire_decode_ms` (required when `wire_format=kvwire_v1`)
- Communication accounting scope: **count only the teacher → receiver payload** (cache blob or text message). Do **not** count the shared task prompt.
- Commit `quantization/data/step_*` to git after each milestone so results are reviewable.
- Cleanup hygiene: remove failed runs that do not contain `results/` or have `status=failed` in manifests (keep only validated runs).
- Registry keys now include **M6 schedule hash**, **M7 alignment flag**, and **M8 scope/sparse‑fuse flags**, plus `wire_format` so estimate vs measured-byte runs don’t overwrite each other.

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
  - `RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 RUN_MILESTONE_5=1 RUN_MILESTONE_6=1 RUN_MILESTONE_7=1 RUN_MILESTONE_8=1 RUN_MILESTONE_9=1 RUN_MILESTONE_10=1 EVAL_SMOKE=1 SMOKE_LIMIT=50 sbatch quantization/submit_milestones.slurm`
  - Optional overrides:  
    - `M5_RECIPE=quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8_smoke.json`  
    - `M6_LAYER_SCHEDULES="quantization/configs/kv_layer_schedule.yaml quantization/configs/kv_layer_schedule_last2.yaml quantization/configs/kv_layer_schedule_last8.yaml"`  
    - `M7_BASE_MODEL=Qwen/Qwen3-0.6B M7_TEACHER_MODEL=meta-llama/Llama-3.2-1B-Instruct M7_ALIGNMENT_MODES="0 1"`  
    - `M8_SELECT_MODES="front vnorm_topk proj_vnorm_topk" M8_SELECT_PROPORTIONS="1.0 0.5 0.25" M8_SELECT_SCOPES="prompt" M8_SPARSE_FUSE_MODES="1"`
    - `M9_SELECT_MODES="delta_proj_vnorm_topk" M9_SELECT_PROPORTIONS="0.25 0.10" M9_SELECT_SCOPES="prompt"`
    - `M10_BUDGETS="1/32 1/16 1/8 1/4" M10_PRECISION_CANDIDATES="drop int4 int8"`

### Full runs (paper‑quality)
- [ ] **All milestones via SLURM (full, skip done by default):**
  - `RUN_FULL=1 RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 RUN_MILESTONE_5=1 RUN_MILESTONE_6=1 RUN_MILESTONE_7=1 RUN_MILESTONE_8=1 RUN_MILESTONE_9=1 RUN_MILESTONE_10=1 sbatch quantization/submit_milestones.slurm`
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
  - **Canonical (measured, M11+)**: once `wire_format=kvwire_v1` exists, the analysis must prefer `bytes_measured_total` (includes indices + scales + headers) and fall back to the estimate above only when measured bytes are unavailable.
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

## Milestone 9: Delta-Selective Transfer (Receiver-Redundancy Token Scoring)
**What**  
Add a new token selection mode that ranks tokens by **marginal update in receiver space**:
`token_select_mode: delta_proj_vnorm_topk`.

**Why**  
M8 importance scores (vnorm/proj_vnorm) do not account for **redundancy with the receiver**. If the projected sharer token is already represented in the receiver KV, transmitting it is wasted bandwidth. Delta scoring should improve **accuracy per byte** at low token budgets.

**How**  
Compute full projection once, then score each token by the mean L2 norm of `(proj_v - base_v)` across batch/head. Select top-k tokens and reuse the existing sparse-fuse path.

**Implementation (code changes)**  
- `rosetta/model/wrapper.py`: in `RosettaModel._project_with_transfer`, add a branch for `delta_proj_vnorm_topk` that:
  - applies the same quantize→dequantize path used in transfer **before scoring** (so selection matches the actual channel);
  - computes `proj_key_full, proj_val_full = projector.forward(source_kv_cache, base_kv_cache)` once;
  - computes scores: `scores = norm((proj_val_full - base_value_cache).float(), dim=-1).mean(dim=(0,1))`;
  - applies `token_select_scope` mask before top-k (prompt/instruction/all);
  - **reuses** `proj_*_full` for fusion (gather selected indices) to avoid a second projection pass;
  - uses existing top-k selection + gather + sparse-fuse/scatter;
  - caches indices + score stats in `transfer_cache` and sorts indices deterministically.
- `rosetta/utils/kv_select.py`: reuse the top-k helper; add guard rails for `token_select_min_tokens` and masked positions.
- `quantization/scripts/run_step1_kv_ptq.py` + configs: expose new mode in CLI/config; ensure manifests include the new selection mode.
- `quantization/scripts/analyze_budget_curve.py`: include delta mode in plots/CSV and account for index/scale overhead in bytes.

**Masking order (avoid confounds)**  
- Apply masks in this order: **scope mask → optional kv_cache_proportion within scope → score → top-k**.  
- Do not combine M3 `kv_cache_proportion` with M9 selection in main tables unless explicitly stated.

**Token minimums (avoid hidden budgets)**  
- For M9/M10 experiments, set `token_select_min_tokens=1` (or a tiny % of prompt length).  
- Use `token_select_min_tokens=64` only for stability/debug runs, not for headline curves.

**Telemetry (M9)**  
- Selection: `selected_tokens`, `total_tokens`, `selected_fraction`, `score_mean/min/max`.  
- Overhead: `projection_score_time_ms`, `selection_time_ms`, `projection_transfer_time_ms`, `fuse_time_ms`, `end_to_end_prefill_ms`.  
- Bytes: payload + index bytes + scale overhead (effective bytes/elem).

**Milestone 9 phases (required)**  
- **M9-P0 (Design lock)**: scope = prompt; proportions = {1.0, 0.5, 0.25, 0.10}; modes = {vnorm_topk, proj_vnorm_topk, delta_proj_vnorm_topk}.  
- **M9-P1 (Implementation)**: add the new mode in `wrapper.py`, expose in configs/CLI, and log telemetry fields.  
- **M9-P2 (Correctness)**:  
  - with `proportion=1.0`, outputs must match M8 (exact answers);  
  - selection stats must report 100% coverage (selected_fraction=1.0).  
- **M9-P3 (Perf/overhead check)**: measure selection + projection overhead on 50 samples; if overhead is too high, restrict scope to prompt/instruction tokens.  
- **M9-P4 (GPU smoke)**: 50 samples at {0.25, 0.10} on OpenBookQA.  
- **M9-P5 (Full GPU grid)**: OpenBookQA + ARC-C for {1.0, 0.5, 0.25, 0.10} with {vnorm, proj_vnorm, delta_proj_vnorm}.  
- **M9-P5b (Very-low budget)**: add p=0.05 for **delta only** on one dataset (OpenBookQA) to show extreme-budget behavior.  
- **M9-P6 (Hetero spot check)**: at least one low-budget point (0.25 or 0.10) on Qwen3 <- Llama3.2 to confirm benefit under heterogeneity.  
- **M9-P7 (Analysis)**: update budget curves + golden summary; report deltas at low budgets.

**M9 status (2026‑02‑01)**  
- **DONE**: P0–P5b (full grid + p=0.05).  
- **PENDING**: P6 hetero spot check.  
- **PENDING**: stability shard/seed runs (p=0.10).  
- **PENDING**: system‑timing sync run for reporting.  
- **PENDING**: extra‑dataset run (gsm8k).

**Expected outcome**  
- Delta scoring should beat vnorm/proj_vnorm at lower budgets (<= 0.25), especially for hetero pairs.

**Notes / risks**  
- Requires full projection to score tokens (extra overhead). If too slow, score only prompt/instruction tokens or subsample tokens when scoring.

---

## Milestone 10: RD-C2C (Token x Precision Rate-Distortion Scheduling)
**What**  
Under a fixed communication budget, jointly decide **which tokens** to transmit and **what precision** (drop/int4/int8) per token.

**Why**  
Selection alone (M9) leaves efficiency on the table. Mixed precision at token level should improve the **accuracy-bytes frontier** beyond fixed-precision selection.

**How (high level)**  
Use M9 delta scores as utility and assign each token to {drop, int4, int8} to meet a **byte budget** using a deterministic greedy allocator. The allocator should be stable under ties and enforce the exact budget (including overhead).

**Implementation (code changes)**  
- Config keys (kv_transfer_config):
  - `token_precision_mode: rd_greedy`
  - `token_precision_candidates: [drop, int4, int8]`
  - `token_precision_budget_bits_per_elem: <float>` (derived; log only)
  - `token_precision_budget_bytes: <float>` (primary budget)
  - `token_precision_calib_n: 64`
  - `token_precision_scope: prompt` (align with token_select_scope by default)
- In `wrapper.py`:
  - compute token scores from M9;
  - **RD-Greedy v0**: sort by Δ score (stable sort by score, then index), fill budget with int8 → int4 → drop;
  - enforce `bytes(I8,int8) + bytes(I4,int4) + index/scale overhead ≤ token_precision_budget_bytes`;
  - project/fuse each group separately:
    - int8 group: kv_quant_scheme=int8
    - int4 group: kv_quant_scheme=int4
    - drop group: receiver cache only (no projection)
  - scatter into full cache; ensure group ordering is deterministic.
- Calibration (lightweight):
  - estimate relative distortion scales for int4/int8 from a small calibration subset; store scalars in manifest so runs are reproducible.
- `analyze_budget_curve.py`:
  - compute effective bits/elem for mixed precision;
  - log group counts and bytes.

**Budget definition (required)**  
- Define `B_total_bytes` as a fraction of **full-transfer bytes** (payload + index + scale).  
- The allocator must satisfy `bytes(I8,int8) + bytes(I4,int4) + overhead ≤ B_total_bytes`.  
- `token_precision_budget_bits_per_elem` is a derived diagnostic, not the primary budget.

**Baseline reduction (required sanity)**  
- RD-C2C with candidates `{drop,int8}` must reduce to M9 delta selection at the equivalent byte budget (up to rounding).

**Projection grouping sanity**  
- Validate that projecting tokens as groups vs projecting all tokens as one group yields negligible differences for `proportion=1.0` (document any discrepancies).

**Telemetry (M10)**  
- Group counts: `tokens_drop`, `tokens_int4`, `tokens_int8`, `effective_bits_per_elem`.  
- Overhead: allocation time, projection time by group.  
- Budget accounting: payload + index bytes + scale overhead.

**Milestone 10 phases (required)**  
- **M10-P0 (Design lock)**: budgets = {1/32, 1/16, 1/8, 1/4} of full-transfer bytes; candidates = {drop,int4,int8}.  
- **M10-P1 (Allocator)**: implement the deterministic greedy solver and log assignment stats + effective bits.  
- **M10-P2 (Local sanity)**: 1–5 samples to validate group assignment + scatter; verify budget accounting equals target.  
- **M10-P3 (GPU smoke)**: 50 samples at 1/8 budget on OpenBookQA.  
- **M10-P4 (Full GPU grid)**: OpenBookQA + ARC-C for {1/32, 1/16, 1/8, 1/4}, compare fixed-int8 vs RD (drop,int4,int8).  
- **M10-P5 (Hetero spot check)**: one budget point at 1/8 on Qwen3 <- Llama3.2.  
- **M10-P6 (Analysis)**: report Pareto frontier vs fixed-int8 selection and update budget curves.
 - **M10-P7 (RD ablation)**: compare candidates {drop,int8} vs {drop,int4,int8} at the same budgets.
**M10 evaluation note**  
- M10 budget runs **ignore** `token_select_proportion`; the allocator chooses group sizes to meet the byte budget.

**M10 status (2026‑02‑01)**  
- **DONE**: P0–P4 (budgets 1/32–1/4 on OpenBookQA + ARC‑C).  
- **PENDING**: P5 hetero spot check.  
- **PENDING**: P7 RD ablation ({drop,int8} vs {drop,int4,int8}).  
- **PENDING**: stability shard/seed runs (budget 1/16).  
- **PENDING**: system‑timing sync run for reporting.  
- **PENDING**: extra‑dataset run (gsm8k).

**Expected outcome**  
- RD-C2C should improve accuracy at the same bytes (or reduce bytes at the same accuracy) compared to fixed-precision selection.

**Notes / risks**  
- Requires stable distortion estimates; if noisy, increase `token_precision_calib_n` or use per-layer calibration.

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

## F. Delta Selection + RD-C2C (Milestones 9–10)
| ID | Model Pair | Dataset | Method | Precision | Token Budget | Train |
|---|---|---|---|---|---|---|
| F1 | Qwen3‑0.6B <- Qwen2.5‑0.5B | OpenBookQA | Delta selection | INT8 | 0.25 | no |
| F2 | Qwen3‑0.6B <- Qwen2.5‑0.5B | ARC‑C | Delta selection | INT8 | 0.25 | no |
| F3 | Qwen3‑0.6B <- Qwen2.5‑0.5B | OpenBookQA | RD‑C2C | mixed (drop/int4/int8) | 1/8 | no |
| F4 | Qwen3‑0.6B <- Qwen2.5‑0.5B | ARC‑C | RD‑C2C | mixed (drop/int4/int8) | 1/8 | no |
| F5 | Qwen3‑0.6B <- Llama3.2‑1B | OpenBookQA | Delta selection | INT8 | 0.25 | no |
| F6 | Qwen3‑0.6B <- Llama3.2‑1B | OpenBookQA | RD‑C2C | mixed (drop/int4/int8) | 1/8 | no |

---

## Workshop vs Main‑Conference Path
- **Workshop**: Milestones 0 → 1 → 2 → 3 → 4 (baseline + PTQ + pruning + budget curve).
- **Main‑conf**: Workshop path + M5 (QAT) + M6 (mixed precision) + **systems measurements** (bandwidth/latency or bit‑packing).  
  - **M7 (heterogeneity)** is required (at least one cross‑family pair with alignment on).  
  - **M8 (token‑level sparse C2C)** is required.  
  - **M9/M10 (delta selection + RD‑C2C)** are required.

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

## Extension Plan: Measured-Byte Semantic Cache Communication at Scale (NeurIPS-ready)

> **Paste-in block:** This section is designed to drop into the existing `quantization/IMPLEMENTATION_PLAN.md` as an extension.
> Focus: **measured on-wire bytes**, **7B+ scaling**, **cross-family heterogeneity**, **head-to-head baselines**, and a **regime map** (when cache beats text and when it doesn’t).
> Optional high-upside add-on: **2-round receiver-driven refinement** *only if a pilot shows a clear win*.

---

### High-level goals

1. **Replace all “estimated bytes” claims with measured on-wire bytes** (including indices + metadata + true INT4 packing).
2. **Validate the full story at 7B+ scale** and on at least one **cross-family heterogeneity pair**.
3. **Run head-to-head comparisons** at *equal measured bytes* vs the closest baseline families:

   * text-only communication under byte caps
   * selective KV sharing baselines (KVComm-like)
   * adaptive mixed-precision KV compression baselines (Q-KVComm-like)
   * existing semantic cache transfer baseline (C2C-style, what we already have)
4. Produce a **regime map**:

   * identify and explain **where semantic cache comm beats text**, and where it **does not**, under strict measured byte budgets.
5. (Optional, gated) Add **receiver-driven refinement** (2-round max) if the pilot clearly improves bytes-to-success / success-at-cap.

---

### Non-goals (to avoid scope creep)

* We do **not** claim the wire format itself is novel; it is **credibility infrastructure**.
* We do **not** build a large new “platform” product. We build a **minimal harness** sufficient for fair comparisons and regime analysis.
* We do **not** pursue a learned proxy model unless/until the core results are already strong.

---

## Definitions and accounting contract (must be consistent everywhere)

### Bytes on wire (measured)

For every method point, we record:

* `bytes_measured_total`: exact number of bytes produced by serialization (`len(blob)`), including:

  * token indices
  * quantized K/V payloads (packed INT4 or INT8)
  * scale/zero metadata
  * headers describing shapes/dtypes/version
  * (if interactive) request messages

We also keep:

* `bytes_estimated_total` (legacy sanity check only)

### Latency accounting (recommended)

For every point, record:

* `wire_encode_ms`
* `wire_decode_ms`
* `receiver_forward_ms` (or end-to-end step time)
* optionally `wire_send_ms` if we measure a socket loopback

### Fairness rules for comparisons

* All comparisons against baselines must be done at **equal `bytes_measured_total`** (not equal p, not equal “compression ratio estimate”).
* The byte cap applies only to the **teacher → receiver payload** (KVWire blob or text message). The shared task prompt is provided to the receiver in all conditions and is **not** counted.
* Prompt templates, decoding settings, and evaluation scoring must be identical across methods within a comparison group.
* If a baseline cannot be implemented faithfully, we implement a **clearly labeled “-like” variant** and document the differences.

---

## Tripwires (early kill-switches to prevent wasted weeks)

* **Tripwire T1 (measured bytes drift):** If measured bytes differ from estimates by >15% in target regimes, halt and fix accounting/format before running large sweeps.
* **Tripwire T2 (scale collapse):** If improvements disappear at 7B+ on the within-family pair, pivot the paper emphasis toward **regime map + limitations** rather than method wins.
* **Tripwire T3 (hetero instability):** If cross-family runs are unstable, freeze alignment choices (fixed layer subset + fixed projector) and reduce tuning knobs; do not chase hyperparameters indefinitely.
* **Tripwire T4 (baseline dominance):** If we cannot beat or match strong baseline families at equal measured bytes in *any* meaningful regime, pivot to a **negative/diagnostic regime-map paper** (“when semantic cache transfer fails under realistic byte budgets and why”).
* **Tripwire T5 (interactive pilot fails):** If a 100-sample pilot shows no clear win, do not build a full interactive protocol.

---

### P0 — Plumbing tasks (required before M11)

* [x] **Add milestone IDs and run scripts**
  * `run_step11_kvwire.py`, `run_step12_scale_hetero.py`, `run_step13_baselines.py`, `run_step14_regime_map.py`, `run_step15_refine_pilot.py`
  * Keep naming consistent with existing `stepX_*` conventions.
* [x] **Update SLURM launcher**
  * Add `RUN_MILESTONE_11`...`RUN_MILESTONE_15` and default them off.
  * Ensure `EVAL_SMOKE` and `SMOKE_LIMIT` flow into these steps.
* [x] **Registry key invariants (no collisions)**
  * Add `wire_format`, `wire_version`, `wire_quant_mode`, `wire_scale_granularity`, `wire_index_dtype`, and (if used) `wire_compression` to the registry key.
* [x] **Manifest schema invariants**
  * Guarantee every new run writes:
    * `bytes_measured_total`, `bytes_measured_breakdown`
    * `wire_encode_ms`, `wire_decode_ms`
    * per-stage timings: `prefill_ms`, `select_ms`, `project_ms`, `fuse_ms`, `end_to_end_ms`
* [x] **Per-sample cap rule**
  * All byte budgets are **per-sample caps** (not dataset-average). Log `budget_cap_bytes`, `budget_actual_bytes`, and `budget_slack_bytes`.
* [x] **Compatibility layer**
  * Update analysis to prefer `bytes_measured_total` when present and fall back to `bytes_estimated_total` / `bytes_estimate`.
* [x] **Glue tasks before large sweeps**
  * Add `quantization/scripts/check_model_access.py` (AutoConfig probe + shape print, fail fast on gated/missing models).
  * Add a manifest schema contract test (assert required fields per run).
  * Add a long-context corpus builder and corpus version pinning.
  * Add a budget-cap enforcement unit test (per-sample byte cap respected).
  * Add a sequential vs simultaneous equivalence test (small model, tight tolerance).

### P0 Design Locks (to unblock M11-M15)

#### M12 Pair Selection (locked)

* large_within:
  * receiver: `Qwen/Qwen3-8B`
  * sharer: `Qwen/Qwen2.5-7B`
* large_hetero:
  * receiver: `Qwen/Qwen3-8B`
  * sharer: `mistralai/Mistral-7B-Instruct-v0.3`
* Add `quantization/scripts/check_model_access.py` to fail fast on gated/missing models and print key shapes.

#### M11 KVWire v1 Defaults (locked)

* indices: absolute, sorted ascending
* index dtype: uint16 when max_index < 65536 else uint32
* scale dtype: fp16
* scale granularity default: per_block where block = (layer, token_index, head_id) vector (separate scale for K and V)
* quant modes: int8 + packed int4
* compression: off by default (no entropy coding); optional zstd only as a secondary measurement mode
* add a golden-blob unit test to prevent schema drift

#### Long-context Definition (locked)

* baseline: default prompt only
* long: append deterministic padding context until 8192 receiver-tokenizer tokens
* padding method: concat chunks from a pinned local corpus (no retrieval); deterministic seed = hash(example_id)
* record `corpus_version` and `longctx_seed` in manifests

#### Text baselines (locked)

* Implement 3 styles:
  * text_raw (deterministic)
  * text_summary_heur (deterministic heuristics)
  * text_summary_llm (sharer-generated; main strong baseline)
* text_summary_llm decoding: do_sample=False, temperature=0, top_p=1; post-trim by UTF-8 bytes to fit budget
* Always count UTF-8 bytes; receiver consumes message with a fixed wrapper prompt
* For Qwen3, disable thinking mode for deterministic greedy baseline generation

#### Storage policy (locked)

* Do not store KVWire blobs by default; store metrics + breakdown + timings.
* Optionally store N=10 sample blobs per config + all failure blobs for debugging.

# Milestone M11 — KVWire v1 (Measured bytes + correct INT4 packing)

### Objective

Replace “fake-quant + byte estimates” with **real serialization** and **measured bytes** for every method.

### Deliverables

* `quantization/kvwire/kvwire_v1.py`
* Roundtrip + correctness tests
* Integration into existing selective-transfer and RD pipelines so all runs emit measured bytes + wire timings.

### Implementation steps

#### M11-P0 — Wire contract lock

* [x] Decide whether indices are **absolute** positions or **delta-coded**; document it.
* [x] Decide quantization metadata layout (per-head / per-block) and how scales are stored.
* [x] Decide whether optional compression is included (keep off by default; add later as ablation).

#### M11-P1 — Blob introspection + debug tooling

* [x] Add `quantization/scripts/kvwire_inspect.py` that prints:
  * section sizes (indices/payload/scales/headers)
  * dtype/shape info
  * checksums (optional)

#### M11-P2 — Per-sample accounting + slack

* [x] For each example, log:
  * `bytes_measured_total`
  * `bytes_measured_breakdown`
  * if budgeted: `budget_cap_bytes`, `budget_slack_bytes`

#### M11-P3 — Bridge drift report

* [x] Add `quantization/scripts/report_byte_drift.py`:
  * compare old estimate vs KVWire measured bytes on a small sample
  * output mean/median/p95 drift by method (INT8/INT4/sparse)

#### M11.1 — Implement KVWire v1 serialization

Create:

* `quantization/kvwire/kvwire_v1.py`

Required API:

* `pack(payload: dict, cfg: KVWireConfig) -> bytes`
* `unpack(blob: bytes, cfg: KVWireConfig) -> dict`

Minimal payload fields:

* `version` (e.g., `kvwire_v1`)
* tensor metadata (layer count, head dims, dtypes, shapes)
* `indices` (positions)
* `k_quant`, `v_quant` (INT8 or packed INT4)
* `scales` (FP16/BF16; granularity configurable)
* optional `zeros` if asymmetric

Config options:

* `wire_index_dtype: uint16|uint32` (uint16 when safe)
* `wire_scale_dtype: fp16|bf16`
* `wire_quant_mode: int8|int4`
* `wire_scale_granularity: per_tensor|per_layer|per_head|per_block`
* `wire_include_headers: bool` (default true)

Design constraints (keep simple):

* Single contiguous blob (header + sections), no entropy coding initially.
* Avoid per-token headers; keep section headers constant-size.

#### M11.2 — Correct INT4 packing/unpacking

Add helpers (same file or `quantization/kvwire/int4.py`):

* `pack_int4(int8_tensor_in_range[-8,7]) -> bytes`
* `unpack_int4(blob, shape) -> int8_tensor`

#### M11.3 — Tests (non-negotiable)

Add:

* `quantization/tests/test_kvwire_roundtrip.py`
* `quantization/tests/test_kvwire_int4.py`

Coverage:

* random tensors -> quantize -> pack -> unpack -> dequantize
* verify shapes/dtypes
* verify INT4 byte length (`ceil(n/2)` with defined padding)
* verify errors within expected quant bounds

#### M11.4 — Integrate KVWire into existing run steps

Wherever current code estimates bytes, add:

* `wire_format: estimate|kvwire_v1` (default estimate during rollout)
* when `kvwire_v1`: quantize -> pack -> record `len(blob)` -> unpack -> dequantize -> apply

Update manifests to include:

* `bytes_measured_total`
* `bytes_measured_breakdown` (indices/payload/scales/headers)
* `wire_encode_ms`, `wire_decode_ms`
* keep `bytes_estimated_total` for sanity

#### M11.5 — Minimal equivalence harness

Add:

* `quantization/scripts/debug_kvwire_equivalence.py`

Run it on a tiny batch to validate:

* pack/unpack path is functionally equivalent (modulo quant noise)
* measured bytes stable across runs

### Acceptance criteria

* KVWire roundtrip tests pass.
* Measured bytes emitted for at least 2–3 existing golden points (one INT8, one INT4, one sparse).
* Measured-vs-estimated drift <15% in target configs (or drift explained and estimates updated).

---

# Milestone M12 — Scale + Heterogeneity Grid (7B+ and cross-family)

### Objective

Make the paper defensible: results at 7B+ and at least one cross-family pair.

### Deliverables

* Pair configs for:

  * a 7B+ within-family pair
  * a 7B+ cross-family (heterogeneous) pair
* Sequential execution mode to run sharer and receiver on a single GPU if needed
* A standardized prompt builder and evaluation scoring pipeline used across all runs

### Implementation steps

#### M12-P0 — Model/pair configs are first-class artifacts

* [x] Create `configs/pairs/*.json` and require a `pair_id` in every run.
* [x] Log `pair_id` and resolved model IDs into manifests.

#### M12-P1 — Sequential mode as a production feature

* [x] Implement `--exec-mode sequential|simultaneous` and ensure:
  * sequential mode saves payloads in `wire_blobs/` with a stable naming scheme
  * resume works if receiver run crashes (do not recompute sharer)

#### M12-P2 — Hetero instability guard rails

* [x] Add a hard alignment on/off flag with a default for hetero pairs.
* [x] Require a smoke run (N=50) before full eval for any new pair/dataset.

#### M12-P3 — Long-context setup

* [x] Define exactly how long-context prompts are created (synthetic concat, retrieval chunks, etc.).
* [x] Log `context_len_bucket` and actual prompt length stats.

#### M12.1 — Define the evaluation grid (explicit table in repo)

Add a table in the plan + a machine-readable config (JSON/YAML) such as:

* `quantization/configs/grids/scale_hetero_grid.json`

Minimum axes:

* `pair_id ∈ {small_within, large_within, large_hetero}`
* `task ∈ {OpenBookQA, ARC-C, +1 non-MC task}`
* `context_len ∈ {baseline, long}`
* `budget ∈ {full, 1/2, 1/4, 1/8, 1/16}` **defined by measured bytes**

#### M12.2 — Pair configuration layer

Create:

* `quantization/configs/pairs/<pair_id>.json`

Include:

* sharer model name
* receiver model name
* tokenizer strategy
* alignment strategy + required mapping files
* max_seq_len and any offload rules

#### M12.3 — Sequential execution mode (single-GPU friendly)

Implement a runner option:

* `--exec-mode simultaneous|sequential`

Sequential mode:

1. Load sharer -> compute KV/representations -> pack KVWire blob(s) -> save to CPU RAM/disk
2. Unload sharer (`del model; torch.cuda.empty_cache()`)
3. Load receiver -> unpack -> apply -> evaluate

Add helper utilities:

* `quantization/utils/model_lifecycle.py` (load/unload/gc wrappers)

Validation:

* On small models, sequential and simultaneous should match (within quant noise).

#### M12.4 — Standardize prompts and scoring

Add:

* `quantization/prompts/builders.py`
* record `prompt_template_id` and `decode_settings_hash` into manifests

Ensure scoring is deterministic (forced-choice scoring for MC where possible).

### Acceptance criteria

* At least one 7B+ within-family run and one 7B+ cross-family run complete with measured bytes.
* Same prompt builder used for all methods on a dataset.
* Sequential mode matches simultaneous on the small baseline pair.

---

# Milestone M13 — Head-to-Head Baselines at Equal Measured Bytes

### Objective

Prevent “non-novel / repeated work” critique by showing you can compete with the closest baseline families at equal measured bytes.

### Deliverables

Baseline implementations:

1. Text-only comm (byte-capped)
2. KVComm-like selective KV sharing baseline
3. Q-KVComm-like adaptive compression baseline
4. Existing semantic cache transfer baseline (what you already have)

### Implementation steps

#### M13-P0 — Baseline IO contract

* [x] Every baseline must produce:
  * `payload` (text string or KVWire blob)
  * `payload_bytes_measured`
  * `encode_ms` / `decode_ms` equivalents
* [x] The receiver must consume baselines through a single standardized interface:
  * `apply_payload_to_receiver(example, payload, mode)`

#### M13-P1 — Text baseline must be byte-capped exactly

* [x] Implement strict enforcement:
  * count UTF-8 bytes
  * truncate safely (do not break JSON / tags if you use structured prompts)
  * log truncation rate
* [x] Add at least two text variants:
  * “teacher hint / short rationale”
  * “teacher extracted facts”

#### M13-P2 — Compute fairness note

* [x] Decide and document whether text message generation uses:
  * the teacher model (recommended) or a fixed heuristic
* [x] Log teacher generation settings used for text messages (temperature, max tokens).

#### M13-P3 — Budget matching

* [x] When plotting “equal bytes,” match by **byte cap** (same cap), not by achieved bytes.
* [x] Report slack distribution (median slack, % hitting cap).

#### M13.1 — Text-only baselines (must be fair + byte-accurate)

Create:

* `quantization/baselines/text_comm.py`

Implement:

* `count_utf8_bytes(text: str) -> int`
* `make_message(example, budget_bytes, style) -> str`

Provide at least two styles:

* `text_raw`: minimal necessary information within budget
* `text_summary`: compressed summary within budget (simple heuristics acceptable)

Ensure receiver consumes this message in a standardized way (same prompt wrapper).

#### M13.2 — KVComm-like baseline (documented “-like” variant)

Create:

* `quantization/baselines/kvcomm_like.py`

Implement one or two transparent heuristics (choose ones you can justify and reproduce):

* recency / back-keep heuristic
* norm-based token selection (you already have vnorm-style; reuse)
* optional: attention proxy if already available cheaply

The baseline must emit KVWire payloads so `bytes_measured_total` is comparable.

#### M13.3 — Q-KVComm-like baseline (layerwise / mixed-precision schedule)

Create:

* `quantization/baselines/qkvcomm_like.py`

Implement:

* a simple layerwise sensitivity schedule that assigns INT4 vs INT8 under a measured byte cap
* optionally: include a “calibration” step for hetero pairs if your pipeline already does it

Again: must emit KVWire payloads.

#### M13.4 — Equal-bytes comparison harness

Update/extend the existing budget-curve generator so that for each run:

* budget is enforced by **measured bytes**
* comparisons across methods are done at matched budgets

### Acceptance criteria

* For each dataset + pair, produce a plot/table comparing:

  * best cache method vs best text baseline vs kvcomm_like vs qkvcomm_like
  * all at equal `bytes_measured_total`
* If no method wins anywhere, trigger pivot to “regime map / limitations” framing.

---

# Milestone M14 — Regime Map + Diagnostics (“When cache beats text, and why”)

### Objective

A paper-grade result that survives even if gains are modest: a clear map of regimes where semantic cache transfer is worthwhile under real byte budgets.

### Deliverables

* A single analysis script that produces:

  * regime heatmap/table (cache > text vs text > cache)
  * representative frontier plots
  * bytes breakdown diagnostics

### Implementation steps

#### M14-P0 — Define “best method in cell”

* [x] Decide tie-breakers:
  * primary: accuracy
  * secondary: lower bytes or lower latency
* [x] Keep this deterministic and log it.

#### M14-P1 — Uncertainty / stability

* [ ] For 2–3 key cells, run:
  * 2 seeds or 2 shards (bootstrap)
* [ ] Plot error bars or include a stability table in appendix.

#### M14-P2 — One-click artifact generation

* [x] Create `make_regime_map.sh` that:
  * ingests runs root
  * outputs plots + CSV tables + a single summary markdown

#### M14.1 — Unified result ingestion

Create:

* `quantization/scripts/analyze_regime_map.py`

It should ingest manifests across all runs and construct a dataframe keyed by:

* pair_id
* dataset/task
* context_len_bucket
* budget_bucket (based on `bytes_measured_total`)
* method_family

#### M14.2 — Compute the regime map

For each cell:

* identify best cache-family method
* identify best text-only baseline
* compute:

  * Δaccuracy at fixed bytes
  * bytes required to hit a fixed accuracy target (inverse view)
  * stability across seeds (where available)

#### M14.3 — Add “why” diagnostics

Per cell/point, compute:

* bytes breakdown: indices vs payload vs metadata
* fraction of tokens kept
* front/back pruning behavior
* which layers dominate bytes (if available)

#### M14.4 — Produce paper-ready artifacts

Outputs:

* `analysis/regime_map/regime_table.csv`
* `analysis/regime_map/heatmap.png`
* `analysis/regime_map/frontiers/<pair>/<task>.png`
* `analysis/regime_map/bytes_breakdown_<pair>_<task>.png`

### Acceptance criteria

* One main-figure regime map exists for the paper.
* Appendix artifacts exist for full grid.
* The map is stable under minor run perturbations (seed/task shard).

---

# Optional Milestone M15 — Receiver-Driven Refinement (2-round max, pilot-gated)

> **Do not implement fully unless the pilot wins.**

### Objective

Try a high-upside “new knob” without derailing the core plan. This should never be the only novelty; it’s additive.

### Pilot (required before full build)

* Run 100 samples on 1–2 tasks, 1 pair, 2 budgets:

  * Round 0: cheap payload
  * Round 1: refinement payload (add indices and/or upgrade precision)
* Decision rule:

  * MC tasks: forced-choice margin trigger
  * numeric tasks: self-consistency / verifier-style check if available
* Count request bytes and latency.

#### M15-P0 — Pilot-only first

* [x] Implement only for:
  * 1 model pair
  * 1 dataset
  * 2 byte caps
  * max 2 rounds

#### M15-P1 — Stop criteria defined upfront

* [x] Use task-structured criteria (MC margin / verifier pass), not generic “confidence.”
* [x] Log:
  * % examples using round 2
  * bytes added by round 2
  * net accuracy change at fixed cap

#### M15-P2 — Hard kill-switch

* [x] If pilot does not improve bytes-to-success or success-at-cap, do not proceed.

### Full implementation only if pilot succeeds

* Hard cap: 2 rounds
* Must report:

  * success-at-cap and/or bytes-to-success improvements
  * latency impact (encode/decode + extra receiver pass)

### Acceptance criteria

* Clear improvement vs best one-shot method in at least one meaningful regime (tight budget or long context), measured in bytes-to-success and/or success-at-cap.
* If not, abandon and do not include except as a short negative appendix note.

---

# Bridge plan: minimum re-sweep after KVWire

Once M11 lands, do this targeted KVWire calibration sweep:

* [ ] Full-transfer measured bytes at INT8 and INT4 on the primary pair for:
  * OpenBookQA + ARC-C
* [ ] One selective-transfer point (e.g., p=0.25) INT8 + INT4
* [ ] One delta-selection point (e.g., p=0.10) INT8
* [ ] One RD point (e.g., 1/16 budget) with `{drop,int8}` and `{drop,int4,int8}`
* [ ] Run `report_byte_drift.py` and decide:
  * drift small -> keep most legacy curves; upgrade future runs to measured bytes
  * drift large -> re-run only the most important frontiers (not the entire grid)

# Execution order (recommended)

1. **M11 KVWire** (measured bytes correctness first)
2. **M12 scale + hetero** (single-GPU sequential mode if needed)
3. **M13 head-to-head baselines** (equal-bytes fairness)
4. **M14 regime map + diagnostics** (paper-grade framing)
5. **M15 optional interactive pilot** (only if the pilot shows a win)

---

# “If results aren’t what we expect” pivot plan (explicit)

* If **measured bytes** reveal overhead dominates (indices/metadata too large):

  * change index dtype (uint16 when safe)
  * change scale granularity (per-block instead of per-head)
  * rerun a small grid before any large sweeps

* If **scale (7B+)** results weaken:

  * reframe contribution around the **regime map** and the scaling diagnosis
  * identify which components fail to scale (quant vs selection vs alignment)

* If **cross-family hetero** is unstable:

  * freeze mapping choices, reduce tuning knobs
  * prefer fewer, cleaner hetero results over many unstable ones

* If **baselines dominate**:

  * pivot to a diagnostic paper: “semantic cache transfer under realistic measured byte budgets often fails; here are the regimes and the reasons”
  * this is still publishable if the analysis is deep, broad, and honest

* If **interactive pilot fails**:

  * drop interactive from mainline scope immediately

---

# Paper-facing checklist (what must exist before writing)

* All headline plots use `bytes_measured_total` (KVWire)
* At least one 7B+ within-family and one 7B+ cross-family set of results
* Head-to-head baseline tables at equal measured bytes
* Regime map figure + diagnostic breakdown plots
* A short “limitations / when text wins” section backed by regime map evidence
