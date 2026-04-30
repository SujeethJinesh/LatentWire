# Serving SLO Envelope References

- date: `2026-04-30`
- purpose: primary-source grounding for the source-private serving SLO
  envelope. This memo explains why the systems table uses TTFT/TPOT/goodput
  language, transfer-granularity accounting, batching, and explicit GPU
  non-claims.

## FlashAttention

- source: https://arxiv.org/abs/2205.14135
- blocker helped: a raw byte count is not enough for a systems-facing paper.
- mechanism idea: report IO/memory movement and hardware hierarchy effects,
  not only arithmetic or semantic payload.
- next experiment change: keep raw payload bytes, 64B line rounding, 128B DMA
  burst rounding, and batch packing columns in the serving SLO envelope.
- role: systems framing / overclaim guard.

## vLLM / PagedAttention

- source: https://arxiv.org/abs/2309.06180
- blocker helped: reviewers will compare LatentWire with serving systems whose
  bottleneck is KV-cache memory and batching, not only model quality.
- mechanism idea: separate packet communication from KV-cache transport and
  explicitly report whether source KV/cache state crosses the boundary.
- next experiment change: keep `source_kv_exposed` and KV byte-floor rows, but
  do not claim to beat vLLM or PagedAttention on native serving throughput.
- role: serving baseline / comparison boundary.

## DistServe

- source: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- blocker helped: the systems claim needs standard serving vocabulary rather
  than local latency anecdotes.
- mechanism idea: split TTFT, TPOT, and goodput/SLO accounting.
- next experiment change: add TTFT margins for `500/750/1000 ms` SLOs and mark
  TPOT/goodput as unmeasured until native GPU serving exists.
- role: metric standard / future NVIDIA gate.

## Orca

- source: https://www.usenix.org/conference/osdi22/presentation/yu
- blocker helped: packet batching should be tied to serving-scheduler
  precedent instead of asserted as a byte trick.
- mechanism idea: iteration-level scheduling and batching make per-request
  serving costs depend on how work is grouped.
- next experiment change: report single-request and batch-64 packet traffic
  separately, and label batch packing as an accounting assumption.
- role: systems framing.

## KIVI

- source: https://arxiv.org/abs/2402.02750
- blocker helped: naive fp16 KV-cache byte floors are too weak as baselines.
- mechanism idea: practical 2-bit asymmetric KV-cache quantization can shrink
  KV movement dramatically.
- next experiment change: keep KIVI/KVQuant-style byte-floor rows as stronger
  comparators and mark them as native-GPU-unrun.
- role: quantized KV baseline.

## TurboQuant

- source: https://arxiv.org/abs/2504.19874
- blocker helped: latest KV/vector quantization makes any "we compress better"
  systems claim risky.
- mechanism idea: randomized transforms plus low-bit quantization are strong
  compression primitives, especially for KV/cache workloads.
- next experiment change: treat TurboQuant as a comparator/threat model, not
  something LatentWire claims to beat on native KV-cache compression.
- role: latest compression threat / baseline framing.

## QJL

- source: https://arxiv.org/abs/2206.14894
- blocker helped: one-bit random projections and JL-style sketches are strong
  mathematical neighbors for compact vector communication.
- mechanism idea: compare packet schemes against low-bit projection/sketch
  baselines when transporting continuous residuals.
- next experiment change: keep QJL rows in compression/system tables and avoid
  selling product-codebook or scalar packets as novel compression alone.
- role: mathematical baseline / ablation.

## Bottom Line

The serving SLO envelope should be read as a systems trace-card, not as a
production serving benchmark. Its paper value is that it tells reviewers what
crosses the source-target boundary, whether private text/KV state is exposed,
how packet traffic rounds under realistic transfer quanta, which TTFT proxy rows
exist, and exactly what remains unmeasured until GPU serving runs are available.
