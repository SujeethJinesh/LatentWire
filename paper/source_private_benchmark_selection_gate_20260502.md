# Source-Private Benchmark Selection Gate, 2026-05-02

## Status

- current paper readiness: COLM workshop is plausible; ICLR full paper is still
  blocked.
- current story: fixed-byte source-private packets are reproducible on
  ARC-Challenge and OpenBookQA, with HellaSwag now framed as a hard
  headroom/negative-ablation benchmark.
- exact blocking gap: the next positive method must be a train-only receiver
  that beats packet-only with paired uncertainty and source-destroying controls.

## What This Gate Does

This gate stops the loop from continuing to tune saturated HellaSwag hidden-code
variants. In plain terms, it asks: among the benchmarks we already ran, where is
there both a real fixed-byte packet win and enough remaining target/source
complementarity to justify training a receiver?

Artifact:
`results/source_private_benchmark_selection_gate_20260502/`

## Results

| Benchmark | Budget | Seeds | Packet | Target | Same-byte text | Lift vs target | Lift vs text | Oracle headroom | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenBookQA test | 3B | 5/5 | 0.378 | 0.276 | 0.350 | +0.102 | +0.028 | +0.164 | receiver candidate |
| ARC-Challenge test | 12B | 5/5 | 0.344 | 0.265 | 0.311 | +0.078 | +0.032 | +0.185 | receiver candidate |
| CommonsenseQA validation | 2B | 0/5 strict | 0.438 | 0.206 | 0.424 | +0.232 | +0.013 | +0.118 | text-saturated diagnostic |

Decision: OpenBookQA test is the next ICLR method surface. It is already the
second public benchmark, passes the packet/text/control selection gate at 3B,
and has enough target-or-packet oracle headroom to test a real receiver.

## What To Cut

Cut HellaSwag receiver-improvement claims from the main positive-method story.
Keep HellaSwag as:

- systems-rate evidence,
- complementarity/headroom evidence,
- negative ablation evidence for hidden-code/common-basis branches.

Do not promote CommonsenseQA yet. The source signal is strong, but same-byte
text is too close to the packet under the strict text-margin rule.

## Next Exact Gate

Run a train-only OpenBookQA receiver/headroom gate using the promoted 3B packet.
Required controls:

- target-only,
- packet-only,
- target-cache-only,
- candidate-only,
- target-derived packet,
- row-shuffled source packet,
- random same-rate packet,
- label-permutation decoder,
- candidate derangement,
- same-byte visible text,
- source-label-copy.

Promotion requires receiver accuracy at least +0.005 over packet-only with
positive paired CI95 low. A failure keeps COLM plausible but leaves ICLR blocked
until a different learned connector or native NVIDIA method branch passes.

## Systems Update

The cross-benchmark systems comparator was regenerated with additional external
baseline rows for Q-KVComm, KIVI, KVQuant, and SGLang/RadixAttention. These are
still threat-model and byte/exposure rows only; native systems claims remain
blocked until vLLM/SGLang/C2C/KVComm/KV quantization baselines run on NVIDIA.
