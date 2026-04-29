# Source-Private ICLR Evidence Bundle

- date: `2026-04-29`
- status: passed Mac-local evidence packaging gate
- result root: `results/source_private_iclr_evidence_bundle_20260429/`

## Current Readiness

The paper is now a stronger scoped positive-method submission package, but not
yet a broad cross-family latent-transfer paper. The defensible story is
source-private, extreme-rate evidence-packet communication with decoder side
information, strict source-destroying controls, and systems byte accounting.

## What This Gate Adds

I added `scripts/build_source_private_iclr_evidence_bundle.py`, which creates a
single reviewer-facing bundle from the current decisive artifacts:

- rate frontier;
- KV/cache byte lower-bound accounting;
- coded-label/protocol stress gate;
- pass/fail ledger;
- endpoint paired-uncertainty summaries;
- final-table/readiness docs;
- `final/MANIFEST.sha256` checksum status.

The bundle writes:

- `iclr_evidence_bundle.json`
- `iclr_evidence_bundle.md`
- `novelty_matrix.csv`
- `contribution_matrix.csv`
- `reproduce_iclr_evidence_bundle.sh`
- `manifest.json`
- `manifest.md`

## Result

- pass gate: `true`
- contribution rows: `5`
- novelty comparisons: `8`
- pass checks: `10/10`

Passed checks:

- all required artifacts exist;
- rate frontier passes;
- matched-byte text stays at target accuracy;
- packet keeps at least `7.0x` byte advantage over query-aware text;
- QJL-style KV/cache lower-bound is above `1000x` packet bytes;
- coded-label risk gate passes;
- composed label+code+order stress passes;
- core and holdout endpoint uncertainty rows pass;
- pass/fail ledger has at least `3` paper-ready rows.

## Contribution Rows

1. Source-private evidence-packet benchmark and controls.
2. Extreme-rate candidate-syndrome packet method.
3. Systems byte/KV-cache accounting frontier.
4. Endpoint paired uncertainty and local target-decoder evidence.
5. Learned receiver / latent-method diagnostics with explicit failure boundary.

## Novelty Matrix

The matrix positions LatentWire against:

- C2C cache-to-cache communication;
- KVComm-style selective KV sharing;
- TurboQuant/vector-KV quantization;
- QJL 1-bit sign sketches;
- prompt/text compression such as LLMLingua;
- Slepian-Wolf / Wyner-Ziv source coding;
- JEPA/diffusion-transformer latent prediction.

The key novelty claim is not that decoder side information or low-bit
compression is new. The claim is the empirical LLM/agent instantiation:
source-private task evidence, an explicit byte cap, target candidate side
information, source-destroying controls, and a far-left systems frontier.

## Remaining ICLR Risks

- Production serving TTFT/TPOT/throughput on NVIDIA GPUs is still missing.
- The headline method remains protocol/candidate-side-information
  communication, not universal semantic latent transfer.
- Simple learned cross-family masked-innovation receivers failed.
- The final paper must show text relay catches up at higher byte budgets to
  avoid unfair-baseline criticism.

## Next Gate

Build a negative-boundary appendix artifact that aggregates the cross-family
failures and oracle headroom, then decide whether to run an `n=500`
composed-only coded-label stress row or move directly to paper revision.
