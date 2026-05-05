# ARC Candidate-Local Source-Evidence Repair Preflight

- created UTC: `2026-05-05T00:17:26Z`
- COLM_v2 readiness effect: useful negative/guardrail row
- ICLR readiness effect: still blocked
- gate: `arc_candidate_local_source_evidence_repair_preflight`
- status: `failed`

## Current Story

The ICLR direction is to distill useful dense C2C-style transfer into sparse,
source-private packets. The row-level token-pool readout failed because erasing
or shuffling source traces made the same decisions. This follow-up preserved
one Qwen2.5 source hidden vector per answer candidate and asked whether
candidate-local source evidence can repair Qwen3 target errors.

## What Ran

Artifacts:

- public-innovation candidate-local run:
  `results/source_private_arc_candidate_local_source_evidence_repair_preflight_20260505_n32_qwen05_to_qwen3/arc_candidate_local_source_evidence_repair_preflight.json`
- raw candidate-local run:
  `results/source_private_arc_candidate_local_source_evidence_repair_preflight_20260505_n32_qwen05_to_qwen3_no_public_innovation/arc_candidate_local_source_evidence_repair_preflight.json`

Both runs used:

- ARC-Challenge validation;
- 16 fit rows and 16 audit-aligned eval rows;
- Qwen2.5-0.5B-Instruct source hidden features on CPU;
- existing Qwen3 target-only and same-byte visible-text audit rows;
- existing Qwen2.5 source-score cache for source-rank/source-score controls;
- strict controls: target-only, public-only readout, zero-source, wrong-row,
  source-row shuffle, same-source-choice wrong-row, candidate-source roll,
  atom shuffle, coefficient shuffle, candidate roll, source-index,
  source-rank, source-score, and same-byte visible text.

## Result

| run | matched | target-only | public readout | same-byte | source-index/rank/score | best strict control | pass |
|---|---:|---:|---:|---:|---:|---|---|
| public-innovation candidate hidden | 0.3750 | 0.2500 | 0.3125 | 0.3750 | 0.4375 | source-index/rank/score | no |
| raw candidate hidden | 0.3125 | 0.2500 | 0.3125 | 0.3750 | 0.4375 | source-index/rank/score | no |

Public-innovation diagnostics:

- matched minus best strict control: `-0.0625`;
- source-only readout: `0.3750`;
- zero-source: `0.3125`;
- wrong-row source: `0.3125`;
- source-row shuffle: `0.3750`;
- candidate-source roll: `0.3125`;
- atom shuffle: `0.3125`;
- coefficient shuffle: `0.1875`;
- dense candidate-source bytes per row: `14,336`;
- hypothetical top-k sparse proxy bytes per row: `72.0`;
- source extraction wall time: about `23.1s`.

## Interpretation

Candidate-local source evidence is stronger than the row-level token-pool
readout, but it still does not pass. It ties same-byte visible text and loses
to source-index/rank/score controls. The public-innovation version is better
than raw hidden features, but not enough to establish source-causal transfer.

This weakens the Mac-local Qwen2.5-to-Qwen3 ARC hidden-evidence branch. It does
not disprove dense-transfer distillation generally; it says that these small
candidate-local hidden readouts do not recover the useful C2C-like effect.

In lay terms: we stopped giving the target one mixed bag of source clues and
instead gave it separate clues for each answer choice. That helped a little,
but not enough. A simpler baseline that just uses the source model's favorite
answer still did better.

## Decision

- `ruled out`: current ARC candidate-local Qwen2.5 hidden readout, with and
  without train-only public innovation, as a source-causal positive preflight.
- `weakened`: Mac-local Qwen2.5-to-Qwen3 ARC hidden-evidence branch.
- `alive`: dense-teacher/C2C-effect distillation as the next qualitatively
  different branch.
- `not promoted`: Sparse Resonance Packets as a broad positive method.

## Next Exact Gate

Run `svamp32_c2c_teacher_sparse_packet_distillation_preflight`: use the frozen
SVAMP32 C2C teacher surface where target-alone is `8/32`, C2C teacher is
`16/32`, and C2C-only target-complementary wins are `10`. Treat C2C as the
dense teacher, train a sparse low-byte source-private packet to recover C2C
teacher deltas, and require recovery of C2C-only wins beyond target-only,
source-alone, text-to-text, zero-source, shuffled-source, label-shuffle,
source-index/rank/score, and same-byte controls.

If this C2C-teacher proxy also fails, the next ICLR path likely needs NVIDIA
hardware for native KV/C2C comparisons or a new benchmark where dense transfer
has clearer complementary headroom.

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_build_source_private_arc_candidate_local_source_evidence_repair_preflight.py
./venv_arm64/bin/python scripts/build_source_private_arc_candidate_local_source_evidence_repair_preflight.py
./venv_arm64/bin/python scripts/build_source_private_arc_candidate_local_source_evidence_repair_preflight.py --no-public-innovation --output-dir results/source_private_arc_candidate_local_source_evidence_repair_preflight_20260505_n32_qwen05_to_qwen3_no_public_innovation
```
