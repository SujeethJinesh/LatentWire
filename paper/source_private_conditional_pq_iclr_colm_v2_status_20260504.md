# Conditional PQ ICLR / COLM_v2 Status

- date: `2026-05-04`
- artifact:
  `results/source_private_conditional_pq_iclr_colm_v2_status_20260504/`
- code:
  `scripts/build_source_private_conditional_pq_iclr_colm_v2_status.py`
- tests:
  `tests/test_build_source_private_conditional_pq_iclr_colm_v2_status.py`
- references:
  `references/739_conditional_pq_iclr_colm_v2_status_refs_20260504.md`
- status: promotes conditional PQ as the current COLM_v2 live positive branch;
  keeps ICLR blocked on broader positive evidence.

## Purpose

The project had multiple competing live branches after the Sparse Resonance
Packet pivot. Recent ARC atom/residual/transcoder probes failed strict controls,
while the older conditional-PQ innovation branch already had same-family n500
passes, less-diagnostic rows, 2-byte rows, cross-family negatives, and a packet
systems waterfall. This memo consolidates those artifacts into one readiness
status so the next turn does not restart from a weaker branch.

Layman version: this is the lab-note checkpoint. It says "this tiny repair-code
method is the best thing we currently have, here is exactly where it works, and
here is exactly where it still fails."

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_conditional_pq_iclr_colm_v2_status.py
```

Validation:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_conditional_pq_iclr_colm_v2_status.py \
  tests/test_run_source_private_conditional_pq_innovation_gate.py \
  tests/test_summarize_source_private_conditional_pq_basis_schema_grid.py
```

## Consolidated Result

The generated status says:

- COLM_v2 readiness: `scoped_positive_method_ready_for_writeup`;
- ICLR readiness: `blocked_by_cross_family_or_broader_benchmark_positive_gate`;
- same-family disjoint n500 rows: `16/16` pass;
- less-diagnostic n500 rows: `8/8` pass;
- budget-2 n500 rows: `4/4` pass;
- original bidirectional cross-family rows: `0/2` pass;
- cross-family basis/schema grid rows: `0/28` pass.

Systems accounting attached to the live rows:

- method payloads: `2` and `4` bytes;
- framed method records: `5` and `7` bytes;
- minimum KV-cache floor row in the local waterfall: `21504` bytes;
- source text exposed: no;
- source KV exposed: no;
- native GPU/HBM/energy claims allowed: no.

## Interpretation

Promote for COLM_v2:

1. A source-private conditional innovation packet that sends
   `source(matched) - source(answer-masked)`.
2. Target/public side-information decoding against candidate innovations.
3. Utility-per-byte and packet-ISA accounting for 2-byte and 4-byte payloads.
4. Strict claim boundary: same-family/shared-schema positive, cross-family
   negative.

Do not promote for ICLR yet:

1. broad unseen-family latent communication;
2. sparse atom/resonance steering as a working method;
3. native GPU, HBM, PCIe/NVLink, energy, or serving throughput claims.

## Next Exact Gate

Promote the public-conditioned conditional-PQ resurrection gate:

```text
n256 bidirectional held-out family first,
then n500/remap repeat only if positive.
```

Method change:

```text
Replace static public bases with target-public conditioned residual/codebook
decoding while keeping the same source-private conditional innovation packet.
```

Pass bar:

```text
source - best destructive/shortcut control >= +0.10
and paired CI95 low versus best control > 0
```

Required controls:

- target-only;
- answer-masked source;
- constrained wrong-row source;
- same-source-choice wrong-row;
- candidate roll or deranged public basis;
- permuted codes;
- random same-byte packet;
- opaque-slot or deranged-basis decoding;
- source-index/rank/score comparators when they are not pure answer oracles;
- same-byte visible text.

## Branch Decision

This weakens further work on the latest sparse atom/residual steering variants
as the immediate next ICLR path. Those branches may return only after they have
a target-native causal lift or a receiver objective that makes corrupted
packets decode to no-op. The current main branch should be conditional
side-information packets, because it is the only branch with repeated n500
same-family positive evidence and a working systems table.
