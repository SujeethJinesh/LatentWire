# HANDOFF: Experimental Project State For GPU-Side Codex

Last updated: 2026-05-07 (revised to add Phase 0 portfolio and GPU swarm orchestration)

Repo: `/Users/sujeethjinesh/Desktop/LatentWire`

Branch at handoff creation: `codex/sinkaware-per-head-readiness`

Latest pushed commit before this handoff file: `f29da91a458605f223ada0c6567ccddfd65c818f`

## Executive Summary

| Project | Current state | Completion | GPU blocked? | Next action |
|---|---:|---:|---|---|
| HybridKernel | killed by Phase 2 profiler gate below shelf/no boundary signal | 0% active | no GPU allowed under killed prereg | preserve artifacts; use diagnostic only for fresh positive-method pivots |
| Decode Microkernel Consolidation | fresh positive-method pivot; Phase 0 PASS and Phase 1 replay PASS; Phase 2 infra failed because no supported vLLM DMC serving hook exists | Phase 2 blocked | yes | implement a real DMC serving integration or fresh positive-method pivot before any paper-level speedup claim |
| ThoughtFlow-FP8 | alive as falsification-methodology paper only; paper-polish gate PASS/buildable | ~90-93% | no | human copyedit/venue-framing review; no new experiments |
| OutlierMigrate | Phase 0 PASS, Phase 1 PASS, and partial Phase 2 PASS on Nemotron-3: migration fraction 0.820809591642925 with CI95 [0.7865325261158594, 0.8544931149097815]; Qwen3.6/Kimi deferred | paper iteration after partial Phase 2 | no, except deferred Qwen3.6/Kimi runtime compatibility | committee review, audit, and human decision on completing deferred cross-validation |
| Residual Migration in Hybrid Reasoners | killed by Phase 1: full residual 95th-percentile clipping drop was 0.08333333333333333 with CI95 [0.0, 0.20833333333333334], above the 0.015 kill threshold | 0% active | no GPU allowed under killed prereg | preserve artifacts; use diagnostic only for fresh positive-method pivots |
| SSM-State Lifecycle Compression | killed by Phase 0: all 36 Mamba layers had KS p<0.01 but 0/36 reached the preregistered 2x median drift threshold | 0% active | no GPU allowed under killed prereg | preserve artifacts; use diagnostic only for fresh positive-method pivots |
| SSM Shape-Conditioned Codec | killed by Phase 0: shape-conditioned 4-bit codec improved reconstruction but missed the preregistered 10% mean relative NMSE reduction shelf | 0% active | no GPU allowed under killed prereg | preserve artifacts; no further pivot without fresh depth-2 preregistration on a calibration-defined surface |
| Cross-Layer Quantization Error Compounding | killed by theoretical gate: bound artifact-complete but loose; ratios >5 at depths 1/10/15 | 0% active | no GPU allowed under killed prereg | preserve artifacts; only fresh positive-method pivots with new preregistration |
| HybridFPGA | back-pocketed for post-COLM, MLSys 2027 target | positioning only | not in this sprint | email Thierry re FPGA hardware access timeline; sketch resource estimate |
| SSQ-LR | killed as active COLM branch | 0% active | no GPU allowed | preserve stop artifacts only |
| HORN | killed as active COLM branch | 0% active | no GPU allowed | preserve stop artifacts only |
| HBSM | killed as active COLM branch | 0% active | no GPU allowed | preserve stop artifacts only |

The portfolio has two layers. HybridKernel and ThoughtFlow-FP8 are the mature
submission-track branches and remain unchanged in status. The new Phase 0
characterization branches (OutlierMigrate, Residual Migration, SSM-State
Lifecycle, Cross-Layer Error Compounding) are parallel low-cost shots run from a
shared Mac dump pass; each has a pre-registered Phase 0 binary gate that decides
whether it promotes to Phase 1 (GPU validation on Granite-4-H-Small or larger).
HybridFPGA is back-pocketed for the post-COLM MLSys 2027 sprint and is not part
of this week's GPU swarm. SSQ-LR, HORN, and HBSM remain killed; do not reopen.

The next discriminative facts are: (a) the HybridKernel native profiler packet
on the GPU node, and (b) the Phase 0 Mac gate results for the three new
characterization branches, which feed into a GPU swarm queue executed via Codex
/goal on a single Blackwell node.

## Non-Negotiable Operating Rules

- Do not SSH from this repo or run commands through SSH. If a remote/GPU machine
  is needed, work from a local checkout on that GPU machine, or write commands
  for the user to run.
- Use a repo-local virtual environment. On the Mac, prefer `./venv_arm64`.
  On a Linux GPU node, create a repo-local GPU venv such as `./.venv_gpu` or
  `./venv_x86_64`; do not install into the global interpreter.
- Use `.debug/` for scratch artifacts that should not be checked in.
- Commit and push at the end of a work session.
- Do not reopen SSQ-LR, HORN, or HBSM without a new preregistration written
  before any new rows are inspected.
- Do not add GPU numbers to any paper unless the corresponding artifact checker
  passes.

Primary standing instructions are in:

- `AGENTS.md`
- `experimental/README.md`
- `experimental/project_status_20260506.md`
- `experimental/native_gpu_handoff_20260506.md`

## Current Truth Sources

Read these first in any new Codex session:

1. `AGENTS.md`
2. `experimental/README.md`
3. `experimental/project_status_20260506.md`
4. `experimental/native_gpu_handoff_20260506.md`
5. `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`
6. `experimental/hybridkernel/phase2/native_run_packet_checklist.md`
7. `experimental/thoughtflow_fp8/paper/reviewer_pack.md`
8. `experimental/KILLED_ssq_lr_cross_model_transfer/README.md`
9. `experimental/KILLED_horn_directional_noise_propagation/README.md`
10. `experimental/KILLED_hbsm_sensitivity_heterogeneity/README.md`

## Phase 0 Characterization Branches (New)

These four branches were added on 2026-05-07 after a multi-round novelty audit
concluded that most software-only quantization angles in 2026 are scooped
within weeks. The surviving openings sit in narrow architectural niches:
outlier behavior in current 2026 hybrid reasoners, age-based SSM state
compression, residual-stream causal importance for hybrids, and theoretical
error compounding across layers. Each branch is intentionally low-cost in Phase
0 (Mac-only, shared dump pass, 3-5 days), pre-registered, and fail-fast.

The four branches share Granite-4.0-H-Tiny activation dumps from a single Mac
pass over 12 AIME-2025 reasoning traces; they fork only at the analysis stage.
This is what makes parallel execution cheap: one dump pass, four downstream
analyses.

### Branch A: OutlierMigrate

Hypothesis: In current 2026 hybrid Mamba-Transformer reasoning models, the
channels carrying high-magnitude activations during decode are not stationary.
Outlier channels migrate across decode positions in long reasoning traces,
which would mean static outlier-protection recipes (which the 2026 quantization
literature mostly assumes) are misspecified for hybrid reasoners.

Why it survives novelty audit: HCP/CHON (Feb 2026) characterizes outlier
dynamics during pretraining, finding training-time outliers converge to static
"hot channels" by late training. Inference-time decode-time channel migration
in deployed models is unaddressed.

Lab fit: ★★★★★ (outlier dynamics is core Tambe lab specialty).

Phase 0 (Mac):

- Use shared activation dump on Granite-4.0-H-Tiny across 12 AIME-2025 traces.
- Identify top-1% high-magnitude channels at decode positions
  `{100, 500, 1K, 5K, 10K}` per layer.
- Compute channel-position migration matrix.

Phase 0 binary gate:

- Pass: ≥5% of top-1% channels migrate >2 positions across the trace, OR <5%
  migrate (negative result also publishable as "static outliers in hybrid
  reasoners: protection recipes validated").
- Kill: 1-5% migration (effect too small to characterize cleanly, ambiguous
  result not worth a paper).

Phase 1 (GPU): scale to Granite-4-H-Small (full-size hybrid), with formal
statistical testing. Phase 2 cross-model validation uses
Nemotron-3-Nano-30B-A3B-BF16 and Qwen3.6-35B-A3B. Pipeline: COLM 2026
characterization paper -> MLSys 2027 dynamic-protection recipe paper.

### Branch B: Residual Migration in Hybrid Reasoners

Hypothesis: "Rethinking the Outlier Distribution in Large Language Models" (May
2025) showed that massive activations propagated through residual streams in
standard transformers do not actually matter for accuracy: clipping them has no
measurable effect. Whether this holds in current 2026 hybrid Mamba-Transformer
reasoners is open. The architectural balance between attention, SSM, and
residual paths is meaningfully different in hybrids; the finding may flip.

Why it survives novelty audit: Rethinking-Outlier covers standard transformers
only. Mamba-PTQ characterizes hybrid outliers but does not test
residual-stream causal importance.

Lab fit: ★★★★★ (residual-stream outlier propagation, lab specialty).

Phase 0 (Mac):

- Use shared activation dump on Granite-4.0-H-Tiny.
- Identify residual-stream tokens >95th percentile at each layer.
- Run ablation: clip those tokens to 95th percentile, re-run inference on
  AIME-2025 traces, measure accuracy drop.

Phase 0 binary gate:

- Pass-positive: AIME-2025 accuracy drop <1.5% (Rethinking-Outlier finding
  replicates for hybrids -> aggressive residual-stream quantization recipe is
  safe -> recipe paper).
- Pass-negative: drop >3% (Rethinking-Outlier rejected for hybrids ->
  architectural finding paper showing hybrids depend on residual outliers more
  than transformers).
- Kill: 1.5-3% drop (ambiguous, not strong enough either way).

Phase 1 (GPU): cross-model validation on Nemotron-3-Nano-30B-A3B-BF16 and
Qwen3.6-35B-A3B, plus mechanistic ablations isolating attention vs SSM
contribution to the effect. Pipeline: COLM 2026 -> MLSys 2027.

### Branch C: SSM-State Lifecycle Compression

Hypothesis: In hybrid Mamba-Transformer models, the recurrent SSM state at the
end of a long reasoning trace is "old"; most active computation happens on
recent state. Older state can be compressed more aggressively than fresh state.
Different from the killed SSQ-LR branch (which used a static recipe per layer);
this is age-based compression as a function of state position in the trace.

Why it survives novelty audit: NVIDIA explicitly notes there is no sub-FP16
recipe for Mamba state. Quamba/MambaQuant family is pure Mamba PTQ at fixed
precision per state element. Lifecycle (age-based) compression of state across
decode positions is unaddressed.

Lab fit: ★★★★ (outlier-aware SSM specialty).

Phase 0 (Mac):

- Use shared activation dump on Granite-4.0-H-Tiny, capturing SSM state at
  decode positions `{100, 500, 1K, 5K, 10K}`.
- For each layer, compute Kolmogorov-Smirnov test between state distributions
  at positions 100 and 10K.
- Compute magnitude drift ratio per layer.

Phase 0 binary gate:

- Pass: KS p<0.01 on at least 50% of layers AND magnitude drift ≥2x between
  positions 100 and 10K.
- Kill: state distributions stable (no lifecycle to exploit; this is a
  different kill from SSQ-LR's, which was about cross-model static recipe
  transfer).

Phase 1 (GPU): age-conditioned state quantization recipe with quality
benchmarks on full-size hybrid models. Pipeline: COLM 2026 characterization ->
MLSys 2027 recipe paper.

Caveat (read carefully): SSM-State Lifecycle is conceptually adjacent to the
killed SSQ-LR branch. The reframe is "compression as a function of state age"
instead of "static mixed-precision recipe per layer." If the Phase 0 gate fails
similarly to SSQ-LR's S3 cross-model gate (effect does not transfer or is too
small), this should be killed cleanly with a fresh KILL manifest, not reopened
as SSQ-LR. The wedge is materially different but the failure modes may rhyme.

### Branch D: Cross-Layer Quantization Error Compounding

Hypothesis: FP4 quantization error accumulates across layers in a
characterizable way. A closed-form upper bound on accumulated error across N
layers, validated against empirical drift, would provide a principled basis for
layer-wise precision allocation (which is currently ad-hoc in the literature).

Why it survives novelty audit: "Bridging the Gap" (Sept 2025) and related
papers sketch block-level error analysis but do not formalize cross-layer
compounding. Most existing work treats per-layer error as independent; the
compounding term is missing.

Lab fit: ★★★★ (theoretical, but lab-coherent quantization-error analysis
style).

This branch is theoretical and Mac-only. No GPU phase. No shared dump pass
dependency. Can be drafted in parallel with the other three Phase 0 branches.

Deliverable:

- Closed-form upper bound `F(N, sigma_block, sigma_outlier, depth_pattern)`.
- Empirical validation: measure BF16-vs-FP4 drift on Granite-4.0-H-Tiny at
  depths `{1, 5, 10, 15}` and check whether the bound is tight to within 2x of
  measured drift.

Phase 0 binary gate:

- Pass: bound is tight to within 2x of measured drift at all four depths.
- Kill: bound is loose (>5x at any depth) or does not track the empirical
  pattern (indicating the theoretical model is wrong).

Pipeline: COLM 2026 theoretical paper. Optional MLSys 2027 follow-up
implementing layer-wise precision allocation guided by the bound.

## GPU Swarm Queue and /goal Orchestration

The GPU node will execute a queue of gates, not a single experiment. The queue
is defined in `swarm/queue.yml` and orchestrated by a Codex `/goal` running on
the GPU node itself. Codex /goal (CLI version 0.128.0+, ChatGPT-auth)
implements a Ralph loop: plan -> act -> test -> review -> iterate,
autonomously continuing until the queue is exhausted or the token budget is
reached.

The queue, in priority order:

1. HybridKernel profiler gate (always first, unconditional). Reads
   `experimental/hybridkernel/phase2/preregister_profiler_gate.md`. Pass =
   recoverable gain ≥3% with CI lower bound >0 across primary, same-family
   controls below 3%, cross-family controls below 3%, and full artifact checker
   passes. Kill = repeated summaries <1% recoverable gain, controls reproduce
   signal, or packet cannot be made artifact-complete.
2. OutlierMigrate Phase 1 (conditional on Mac Phase 0 pass). Reads
   `experimental/outlier_migrate/phase1/preregister_om_phase1.md`. Phase 1
   gate criteria are in that prereg file; they are stricter than Phase 0 and
   require effect replication on Granite-4-H-Small.
3. Residual Migration Phase 1 (conditional on Mac Phase 0 pass). Reads
   `experimental/residual_migration/phase1/preregister_rm_phase1.md`. Phase 1
   validates the Phase 0 finding on a larger hybrid model with formal
   cross-trace statistics.
4. SSM-State Lifecycle Phase 1 (conditional on Mac Phase 0 pass). Reads
   `experimental/ssm_lifecycle/phase1/preregister_ssml_phase1.md`. Phase 1
   extends KS testing to longer traces (50K positions) and adds magnitude-drift
   quantification.
5. HybridKernel kernel prototype (conditional on entry 1 passing). Triton
   implementation matching profiler-predicted gain within ±20%.
6. Cross-model hybrid validation (conditional on any of entries 2-4 passing).
   Nemotron-3-Nano-30B-A3B-BF16 and Qwen3.6-35B-A3B.

The orchestrator is a single Codex `/goal` issued on the GPU node inside a tmux
session:

```text
/goal "Execute the COLM 2026 GPU swarm queue defined in swarm/queue.yml on this Blackwell node..." --budget 8000000
```

The full /goal text and stop rules are in `swarm/goal_prompt_20260507.md` (to
be created alongside queue.yml and state.json before /goal is issued; see
"Required New Files" below).

The /goal scope explicitly forbids modifying:

- `experimental/preregistrations/`
- `experimental/hybridkernel/phase2/preregister_*`
- `experimental/<phase 0 branch>/phase0/preregister_*`
- `experimental/<phase 0 branch>/phase1/preregister_*`
- `experimental/thoughtflow_fp8/`
- any paper drafts
- any `KILLED_*` manifests

The /goal scope permits modifying:

- `experimental/shared/results/`
- `swarm/queue.yml` (only to update conditional-promotion entries)
- `swarm/state.json` (run summary, GPU-hours used, next action)
- `experimental/<project>/phase1/results/`

Stop rules for /goal:

- Three consecutive `FAIL_INFRA` results in a row -> pause goal.
- Any preregistration file modified during execution -> pause goal
  (`audit_swarm_completion.py` checks git diff against starting SHA).
- GPU wall-time exceeds 140 hours -> graceful wrap-up.
- Token budget exhausted -> enter budget-limited state, produce final
  `state.json` with summary.

False-completion mitigation: /goal must not mark a gate "achieved" unless the
corresponding checker exits with exit code 0 AND the expected decision string.
`audit_swarm_completion.py` re-runs every checker at the end; the goal is not
truly complete until the final audit passes.

## HybridKernel: Live Positive-Method Branch

### Status

HybridKernel is the only live positive-method branch. It is not a result yet.
The project asks whether attention/SSM boundaries in hybrid models create a
separable conversion, materialization, launch, or locality overhead that could
support a boundary-fusion systems paper.

Current completion: approximately 70% if the GPU gate passes, but 0% as
evidence without native GPU profiling.

### Key Files

Paper/reviewer context:

- `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf`
- `experimental/hybridkernel/paper/hybridkernel_colm2026.tex`
- `experimental/hybridkernel/paper/reviewer_pack.md`
- `experimental/hybridkernel/README.md`
- `experimental/hybridkernel/progress.md`

GPU run documents:

- `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`
- `experimental/hybridkernel/phase2/native_run_packet_checklist.md`
- `experimental/hybridkernel/phase2/native_control_matrix.json`
- `experimental/hybridkernel/phase2/reduction_worksheet_template.tsv`
- `experimental/hybridkernel/phase2/cross_family_control_replacement_template.json`

Run/validation code:

- `experimental/hybridkernel/phase2/create_native_run_packet.py`
- `experimental/hybridkernel/phase2/profiler_driver.py`
- `experimental/hybridkernel/phase2/analyze_profiler_metrics.py`
- `experimental/hybridkernel/phase2/check_profiler_run_artifacts.py`
- `experimental/hybridkernel/phase2/check_quality_smoke_artifacts.py`

Tests:

- `experimental/hybridkernel/phase2/tests/test_create_native_run_packet.py`
- `experimental/hybridkernel/phase2/tests/test_profiler_driver.py`
- `experimental/hybridkernel/phase2/tests/test_analyze_profiler_metrics.py`
- `experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py`

### Pre-GPU Local Preflight

On the local checkout before spending GPU minutes:

```bash
cd /path/to/LatentWire
./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase2/tests/test_create_native_run_packet.py \
  experimental/hybridkernel/phase2/tests/test_profiler_driver.py \
  experimental/hybridkernel/phase2/tests/test_analyze_profiler_metrics.py \
  experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py \
  -q
```

Expected status from the previous hardening pass: focused HybridKernel and
related synthetic-gate tests passed. A full repo suite had unrelated top-level
LatentWire ICLR evidence-bundle failures; do not fix those while working this
experimental handoff unless the user asks.

### GPU Node Setup Summary

Follow the runbook exactly. These are the core anchors, not a substitute for
the runbook:

```bash
cd /path/to/LatentWire
python3 -m venv ./.venv_gpu
source ./.venv_gpu/bin/activate
python -m pip install --upgrade pip

export HWK_ROOT="$PWD/experimental/hybridkernel"
export GRANITE_MODEL=ibm-granite/granite-4.0-h-tiny
export QWEN_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
export PREREGISTERED_CROSS_FAMILY_MODEL=

python "$HWK_ROOT/phase2/create_native_run_packet.py" \
  --label granite_boundary \
  --model "${GRANITE_MODEL:?set GRANITE_MODEL before creating the packet}"
```

The create-packet command prints a `run_dir`. Export it:

```bash
export HWK_RUN=/path/printed/by/create_native_run_packet
```

Then fill the packet according to:

- `experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`
- `experimental/hybridkernel/phase2/native_run_packet_checklist.md`

### Required GPU Packet

Promotion requires a complete packet directory, not screenshots or pasted logs.
Minimum admissible packet:

- `metadata/environment.txt` and `metadata/environment.json`
- `metadata/profile_scope.json`
- `metadata/model_provenance.json`
- copied `metadata/native_control_matrix.json`
- `metadata/reduction_input_manifest.json`
- filled reduction worksheet or equivalent cited source file
- row-specific client replay logs
- server-side Nsight Systems logs and artifacts
- server-side Nsight Compute logs and artifacts unless using explicit
  no-boundary-signal kill mode
- `profiler_metrics.json`
- `profiler_analysis_gate.json`
- `profiler_analysis_gate.md`
- `artifact_check.json`

`profiler_metrics.json` must include at least nine valid reduced rows:

- three primary HybridKernel rows
- three same-shape same-family controls
- three same-shape cross-family falsification rows

Rows must have distinct run IDs/artifacts/time windows and must be tied back to
Nsight artifacts via SHA-256 hashes.

### GPU Validation Commands

After trace reduction on the GPU host:

```bash
python experimental/hybridkernel/phase2/analyze_profiler_metrics.py \
  --input "$HWK_RUN/profiler_metrics.json" \
  --output "$HWK_RUN/profiler_analysis_gate.json"

python experimental/hybridkernel/phase2/check_profiler_run_artifacts.py \
  --run-dir "$HWK_RUN" \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

Do not cite or interpret the packet until both commands pass.

### HybridKernel Decision Rule

Promote only if:

- repeated primary recoverable gain clears the preregistered `>=3%` gate;
- bootstrap/interval readout is above zero;
- same-family controls stay below the 3% gate;
- cross-family falsification rows stay below the 3% gate;
- the full artifact checker passes with `--require-full-matrix`.

Kill or shelve if:

- repeated native summaries show less than 1% recoverable gain;
- controls reproduce the same signal;
- the packet cannot be made artifact-complete;
- Qwen/cross-family substitution is done post-hoc rather than through the
  checked-in replacement template before profiling.

If the gate promotes, the next phase is prototype-kernel investigation. Any
prototype speed table must also pass:

```bash
./venv_arm64/bin/python -m experimental.hybridkernel.phase2.check_quality_smoke_artifacts \
  "$QUALITY_SMOKE_JSON" --repo-root "$PWD"
```

## ThoughtFlow-FP8: Alive Paper-Only Falsification Branch

### Status

ThoughtFlow-FP8 is alive only as a falsification-methodology paper. It is not a
positive sparse-cache or FP8 method. Completion is approximately 90-93%.

Do not run a fifth signal. Do not broaden it with SSQ-LR/HORN/HBSM kills.

### Key Files

- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`
- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`
- `experimental/thoughtflow_fp8/paper/reviewer_pack.md`
- `experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`
- `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/README.md`
- `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/manifest.json`
- `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/falsification_table.md`
- `experimental/KILLED_thoughtflow_fp8_positive_method/README.md`

### Current Claim

ThoughtFlow contributes a repo-local registered falsification ladder for
training-free sparse-KV retention signals on reasoning traces. RDU/PSI/VWAC
successor signals failed reproduction or fresh-surface checks. The paper is
valuable as methodology, not as a positive method.

### Current Next Work

Mac-only:

- copyedit;
- venue framing;
- final human review;
- keep historical phase-doc supersession clear;
- no new experiments.

Owned test command:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

The PDF was rebuilt in the previous session after C2C/provenance clarification.

## Killed Branches

Killed means stopped with preserved audit artifacts, not deleted.

### SSQ-LR

Marker:

- `experimental/KILLED_ssq_lr_cross_model_transfer/README.md`

Source project:

- `experimental/ssq_lr/`

Primary stop manifest:

- `experimental/ssq_lr/phase2/s3_transfer_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/ssq_lr/paper/reviewer_pack.md`

Reason killed:

- frozen `mixed_int3_mxfp4_low_error_25pct` recipe on layers `0,30` failed
  no-retuning transfer to Granite 350M;
- layer-0 mixed25/INT3 rescue diagnostics also failed two-model S3.

Do not GPU-promote. Revival requires new preregistration and a fresh surface.

### HORN

Marker:

- `experimental/KILLED_horn_directional_noise_propagation/README.md`

Source project:

- `experimental/horn/`

Primary stop manifest:

- `experimental/horn/phase2/h2_noise_replay_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/horn/paper/reviewer_pack.md`

Reason killed:

- H2 directional drift ratio `1.037`;
- signed selected-direction lower bound `0.324`;
- support `0.5`;
- paired units `6/6`;
- hook-off max delta `0.0`.

Do not GPU-promote. Revival requires a new preregistered full H2/H3 scope and a
concrete reason the near-null Granite Tiny scouts should reverse.

### HBSM

Marker:

- `experimental/KILLED_hbsm_sensitivity_heterogeneity/README.md`

Source project:

- `experimental/hbsm/`

Primary stop manifest:

- `experimental/hbsm/phase2/b1_prompt2_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/hbsm/paper/reviewer_pack.md`

Reason killed:

- two-prompt B1 Fisher p `1.0`;
- boundary top-decile count `0`;
- non-boundary top-decile count `1`;
- cheap-predictor Spearman `-0.667`;
- one-prompt smoke also went wrong direction with Spearman `-0.476`.

Do not GPU-promote. Revival requires a new preregistered narrower mechanism
hypothesis and a fresh surface.

## Reviewer Upload Folder

A reviewer upload folder was created outside the repo at:

- `/Users/sujeethjinesh/Desktop/reviewer_upload_20260507`

It contains the 10 selected reviewer files. It is not tracked in git and will
not exist on a GPU node unless copied separately.

## Useful Test Commands

Focused experimental docs/code smoke used in the previous session:

```bash
PYTHONPATH="$PWD" ./venv_arm64/bin/python -m pytest \
  experimental/ssq_lr/phase2/tests/test_ssq_lr_synthetic_s1_gate.py \
  experimental/horn/phase2/tests/test_horn_synthetic_h1_gate.py \
  experimental/hbsm/phase2/tests/test_hbsm_synthetic_b1_gate.py \
  experimental/hybridkernel/phase2/tests/test_check_profiler_run_artifacts.py \
  -q -x
```

ThoughtFlow saved artifact tests:

```bash
PYTHONPATH="$PWD" ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests/test_saved_falsification_artifacts.py \
  -q -x
```

Full repo suite from the previous session produced `1516 passed, 4 failed`.
The four failures were unrelated top-level LatentWire ICLR evidence-bundle
tests, outside the current five experimental projects. Do not burn time on them
while executing the GPU handoff unless the user asks.

## Required New Files

These files do not yet exist in the repo and must be authored by the human (not
by Codex /goal) before the GPU swarm starts. The preregistration files in
particular constitute the contract that /goal will be measured against; if
/goal authors them, the falsifiability claim collapses.

Swarm orchestration:

- `swarm/queue.yml` (priority-ordered queue with `on_pass`/`on_kill` rules,
  expected GPU-hours per entry)
- `swarm/state.json` (initial state, `gpu_hours_budget = 140`,
  `consecutive_infra_failures = 0`)
- `swarm/audit_swarm_completion.py` (verifiable termination predicate; returns
  exit 0 only when `state.status = COMPLETE` AND every queue entry has a result
  packet AND no preregistration files have been modified since `started_at`
  SHA)
- `swarm/goal_prompt_20260507.md` (the full /goal text including
  scope/stop-rules/budget)

Phase 0 preregistration files (one per branch):

- `experimental/outlier_migrate/phase0/preregister_om_phase0.md`
- `experimental/residual_migration/phase0/preregister_rm_phase0.md`
- `experimental/ssm_lifecycle/phase0/preregister_ssml_phase0.md`
- `experimental/cross_layer_error/preregister_cle_theoretical.md`

Phase 1 preregistration files (one per branch that has a GPU phase):

- `experimental/outlier_migrate/phase1/preregister_om_phase1.md`
- `experimental/residual_migration/phase1/preregister_rm_phase1.md`
- `experimental/ssm_lifecycle/phase1/preregister_ssml_phase1.md`

Shared infrastructure:

- `experimental/shared/unified_dump_pass.py` (single Granite-4.0-H-Tiny pass
  over 12 AIME-2025 traces, dumps activations, residual streams, and SSM state
  at checkpoint positions; cached output feeds all four Phase 0 branches)
- `experimental/shared/check_gate_packet.py` (generic gate checker invoked by
  /goal; reads result packet, returns structured decision)
- `experimental/shared/dump_outputs/` (cache directory, gitignored)

Each preregistration file must specify, at minimum:

- exact gate criteria with numeric thresholds
- exact prompt set with SHA-256
- exact expected decision strings (e.g.,
  `PASS_REAL_OM_DECODE_TIME_MIGRATION`,
  `FAIL_REAL_OM_AMBIGUOUS_EFFECT_SIZE`)
- forbidden inputs (e.g., must not condition on results from other Phase 0
  branches)
- which models are in scope
- which prompts are in scope

The HybridKernel preregistration already exists at
`experimental/hybridkernel/phase2/preregister_profiler_gate.md` and must not be
modified.

## Final Checklist For The Next Codex Instance

1. Start by reading `AGENTS.md`, this `HANDOFF.md`, and
   `experimental/native_gpu_handoff_20260506.md`.
2. Confirm `git status --short` is clean.
3. Confirm whether you are the GPU-side Codex or the Mac-side Codex. The
   current working directory and presence of `nvidia-smi` are reliable
   indicators.
4. If on the GPU node:
   a. Confirm node is Blackwell (`sm_120` or `sm_100`), 96GB+ VRAM, CUDA 12.8
      or newer.
   b. Confirm Codex CLI 0.128.0+ is installed and authenticated against ChatGPT
      (not API key).
   c. Confirm `swarm/queue.yml`, `swarm/state.json`,
      `swarm/audit_swarm_completion.py`, and `swarm/goal_prompt_20260507.md`
      exist and have not been modified since the human authored them.
   d. Confirm all preregistration files listed under "Required New Files" exist
      and have not been modified.
   e. Manually smoke-test one queue entry (typically the HybridKernel profiler
      gate) end-to-end before issuing /goal.
   f. Issue /goal with the prompt from `swarm/goal_prompt_20260507.md`.
   g. Monitor `swarm/state.json` every 4-8 hours; intervene only on consecutive
      infra failures or budget exhaustion.
5. If on the Mac (no GPU):
   a. ThoughtFlow polish only (copyedit, venue framing, human review).
   b. Phase 0 Mac gates for OutlierMigrate, Residual Migration, and
      SSM-State Lifecycle, using `experimental/shared/unified_dump_pass.py`
      once it exists. Run the dump pass once; run the three downstream analyses
      in parallel.
   c. Theoretical work on Cross-Layer Error Compounding (no shared dump
      dependency; can run any time).
   d. Email/coordinate with Thierry re HybridFPGA FPGA hardware timeline
      (post-COLM positioning only; not in this sprint).
6. Do not reopen SSQ-LR, HORN, or HBSM without a new preregistration written
   before any new rows are inspected.
7. Do not add GPU numbers to any paper unless the corresponding artifact
   checker passes.
8. Do not modify any file in `experimental/preregistrations/`,
   `experimental/KILLED_*/`, any Phase 0/Phase 1 preregistration file, or any
   paper draft during the swarm.
