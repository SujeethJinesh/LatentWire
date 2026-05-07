# Local Readiness Recheck

- date: 2026-05-07
- scope: HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8
- purpose: verify that the Mac-side packet validators, paper-boundary tests,
  and Triton-interpreter correctness gates are clean before interpreting any
  native 5090 evidence.

## Result

```text
317 passed, 1 skipped, 2 warnings in 7.84s
```

The skipped test is the opt-in non-interpreter Triton CPU-backend check for the
HybridKernel toy primitive:

```text
experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py
set HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 to run this opt-in gate
```

That opt-in gate was run separately outside interpreter mode on the same Mac
install and passed:

```text
1 passed in 1.35s
```

Command:

```bash
TRITON_CPU_BACKEND=1 HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 \
TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py \
  -q -rs
```

The warnings are import-time `SwigPyPacked` / `SwigPyObject` deprecation
warnings and do not affect the project gates.

## Command

Run from the repository root:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/tests \
  experimental/hybridkernel/phase0/tests \
  experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests \
  experimental/hybridkernel/phase4/tests \
  experimental/ssq_lr/phase2/tests \
  experimental/horn/phase2/tests \
  experimental/hbsm/phase2/tests \
  experimental/shared/tests \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -q -rs
```

## Interpretation

This recheck is not model evidence. It only says the local validators, packet
contracts, paper-boundary tests, and Triton-interpreter correctness checks are
ready to evaluate the next admissible artifacts:

After the paper edits, the HORN and ThoughtFlow-FP8 PDFs were rebuilt with
`latexmk -pdf -interaction=nonstopmode -halt-on-error`.

Follow-up hardening on 2026-05-07 added six local checks without changing the
project decisions: HybridKernel analyzer recovery is capped at the
pre-registered `0.60`, replacement cross-family controls must carry filled
preregistration metadata and matching architecture-map hashes, the future
prototype quality-smoke checker recomputes answer mismatches and output-length
drift from JSONL outputs, SSQ-LR/HORN/HBSM synthetic gates explicitly assert
non-promoting packet decisions, and the ThoughtFlow diagnostic packet now
requires every hashed input path to be tracked for clean-checkout replay.

Second follow-up hardening on 2026-05-07 closed reviewer-found Mac-side gaps:
HybridKernel submitted/native control matrices now require a global
`request_shape`, metric rows must match that shape, and
`profiler_analysis_gate.md` must exactly match the recomputed profiler analysis
rather than only carrying the same status string. The GPU quickstart now writes
`environment_freeze.txt` into `$HWK_RUN/metadata`, and the paper scaffold uses
the strict `--require-full-matrix` artifact-check command. SSQ-LR/HORN/HBSM
schema-rehearsal summaries now carry explicit `evidence_kind:
schema_rehearsal` and `promotable: false` flags, and the shared gate checker
rejects synthetic packets that omit or contradict those fields. ThoughtFlow
PSI/VWAC fresh-surface scripts now accept and record the pinned model/tokenizer
revision used by the runnable replay path, and the paper/diagnostic packet hash
readouts were aligned to the current preregistration hash.

- HybridKernel: native 5090 Nsight/vLLM full-matrix packet; if a prototype is
  later implemented, its quality smoke must pass
  `experimental.hybridkernel.phase2.check_quality_smoke_artifacts` before any
  speed table is cited.
- SSQ-LR: a newly preregistered Mac rescue recipe/layer rule, if pursued.
- HORN: no work unless a new full H2/H3 preregistration exists.
- HBSM: no work unless a new narrower mechanism preregistration exists.
- ThoughtFlow-FP8: paper-only copyedit/human review.
