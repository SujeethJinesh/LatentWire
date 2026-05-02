# Source-Private ICLR/COLM Readiness Update

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets, public-benchmark
  ARC/OpenBookQA endpoints, and byte/exposure systems accounting form the
  defensible core. HellaSwag is no longer a current receiver-improvement
  headline; it is diagnostic/headroom and negative-ablation evidence.
- Exact gap: one positive learned receiver or common-language connector must
  beat packet-only under paired uncertainty, destructive controls, and at least
  one strict cross-family or cross-benchmark falsification.

## Contributions To Put Forward

1. Fixed-byte source-private evidence packets with source-destroying controls.
   Current support: ARC-Challenge and OpenBookQA public-basis rows remain
   positive, and HellaSwag shows strong packet/headroom diagnostics.
2. Systems byte/exposure accounting for packet transfer versus KV/cache
   transfer. Current support: Mac-local decode/packet-ring traces and
   cross-benchmark byte-floor comparators; native vLLM/SGLang throughput rows
   are still pending.
3. A rigorous falsification ladder for latent/hidden-code communication.
   Current support: HellaSwag source-score, receiver-selector, discrete-query,
   anchor-relative common-basis, switch-observability, and PQ hidden-code gates
   are now explicitly recorded as negative or non-headline.

## What Changed

The 2026-05-02 evidence bundle now ingests the HellaSwag PQ hidden innovation
codec gate:

- artifact:
  `results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/hellaswag_pq_hidden_innovation_codec_gate.json`
- default `pq_pca16_m4_k2_identity`: `0.497070` accuracy versus packet-only
  `0.501953`, delta `-0.004883`, CI95 low `-0.017578`;
- best diagnostic scout `pq_pca8_m2_k8_orthogonal`: `0.508789`, delta
  `+0.006836`, CI95 low `0.000000`;
- pass gate: `False`.

Lay explanation: the experiment tried to let TinyLlama send a tiny compressed
fingerprint of its hidden-state reasoning, not just its answer choice. Qwen got
that one-byte fingerprint and its own scores, then tried to choose better. The
fingerprint was sensitive to corruption controls, but it did not reliably help
Qwen beyond the compact packet-only baseline.

## Reviewer-Safe Decision

Cut the claim that HellaSwag currently demonstrates receiver improvement over
packet-only. Keep HellaSwag as:

- a hard non-science benchmark and complementarity/headroom diagnostic;
- a systems byte/exposure row under the fixed packet contract;
- a negative-ablation suite showing which hidden/source-code variants fail.

Do not describe the current HellaSwag branch as robust cross-model latent
reasoning. The stronger claim requires a learned connector that survives the
same controls.

## ICLR Needs

- Positive learned receiver/common-basis method on at least two public
  benchmarks or one benchmark plus a strict cross-family pair.
- Seed repeats, larger frozen slices, paired CIs, and source-destroy controls.
- Direct competitor comparisons against C2C/KVComm-style KV/cache transfer and
  KV quantization byte floors.
- Native NVIDIA serving rows for TTFT, TPOT, goodput, GPU memory, HBM traffic,
  payload bytes, and source-exposure flags.

## COLM Workshop Needs

- Keep the scope honest: source-private packet protocol, public-benchmark
  positives, systems/accounting, and negative latent-code ladder.
- Make the three contributions explicit and avoid claiming solved latent
  reasoning.
- Add one clean figure summarizing the gate tree: ARC/OpenBookQA positive,
  SciQ/CommonsenseQA diagnostic, HellaSwag branch killed for receiver
  improvement.

## Blockers For User Help

- NVIDIA access for vLLM/SGLang native systems baselines and any true
  continuous query/cache connector.
- Confirmation on whether COLM should be framed as a workshop paper with
  negative HellaSwag ablations or held until a positive learned connector
  exists.

## Next Exact Gate

Run a small, local benchmark-selection gate before more HellaSwag work:
identify one benchmark where compact packet-only does not already saturate
source information, then test the current public-basis packet plus packet-only,
same-byte text, target-only, and source-destroy controls. If no such local
surface appears, the next method gate should move to a true learned
query/cache connector on NVIDIA.
