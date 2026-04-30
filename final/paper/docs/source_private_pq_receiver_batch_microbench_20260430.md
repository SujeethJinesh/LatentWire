# Source-Private PQ Receiver Batch Microbench

Date: 2026-04-30

## Status

Current readiness: stronger scoped positive-method paper, still not
comfortably ICLR-full. COLM workshop readiness is strong.

Current paper story: source-private residual communication with decoder side
information. A source emits a 4-byte PQ residual packet; the target uses public
candidate state to decode the packet without receiving source text or source
KV/cache.

Exact blocker addressed here: after the product-codebook model-mediated PQ
receiver failed, the strongest Mac-local systems question became whether the
geometry-mitigated deterministic PQ receiver can be made reviewer-readable as a
small, exact, batched receiver kernel. This gate measures the resident
target-side table lookup across canonical, OPQ, and protected-Hadamard PQ
variants.

Layman explanation: the source sends a tiny code. The target has already built
a public lookup table for the candidate answers. This experiment asks whether
the target can read the tiny code and pick the same answer very cheaply, even
when requests are batched.

## Gate

- script:
  `scripts/build_source_private_pq_receiver_batch_microbench.py`
- test:
  `tests/test_build_source_private_pq_receiver_batch_microbench.py`
- artifact:
  `results/source_private_pq_receiver_batch_microbench_20260430/`
- references:
  `references/540_pq_receiver_batch_microbench_refs_20260430.md`
- setup: n500 eval, 768 train examples, remap seeds `101/103/107`,
  4-byte packets, 3-byte packet-record overhead, `feature_dim=512`,
  slot candidate view, OPQ iterations `4`, batch sizes `1/8/64/256`
- variants:
  - canonical PQ
  - utility-balanced PQ
  - OPQ-Procrustes
  - utility OPQ-Procrustes
  - protected Hadamard PQ
  - utility-protected Hadamard PQ

## Pass Rule

Every remap/variant row must:

- exactly match the canonical geometry decoder under resident table lookup,
- exactly match that same decoder under every batch kernel,
- preserve identical prediction hashes across batch sizes,
- keep resident table p50 and every batch p95 below `0.25 ms/request`,
- and amortize the 7-byte packet record down to 7 bytes/request at the largest
  128B burst batch.

Codebook fitting, packet encoding, and public table construction are reported
separately. They are not claimed as per-token model-speedup measurements.

## Headline Results

The gate passes: `18/18` rows pass.

- max resident table decode p50: `0.0167 ms/request`
- max batch-64 p50: `0.0163 ms/request`
- max batch mismatch count: `0`
- max resident-table mismatch count: `0`
- prediction hashes are invariant across batch sizes for every row
- raw payload bytes/request: `4`
- packet record bytes/request: `7`
- batch-256 128B raw-payload bytes/request: `4.00`
- batch-256 128B packet-record bytes/request: `7.00`
- public table build p50 max: `1.9629 ms`

Representative rows:

| Remap | Variant | Accuracy | Resident p50 ms | Batch64 p50 ms | Mismatches | Batch256 record bytes/request |
|---:|---|---:|---:|---:|---:|---:|
| 101 | canonical | 0.482 | 0.01662 | 0.01623 | 0/0 | 7.00 |
| 101 | utility-protected Hadamard | 0.504 | 0.01642 | 0.01612 | 0/0 | 7.00 |
| 103 | protected Hadamard | 0.514 | 0.01646 | 0.01613 | 0/0 | 7.00 |
| 107 | canonical | 0.520 | 0.01650 | 0.01620 | 0/0 | 7.00 |
| 107 | utility-protected Hadamard | 0.516 | 0.01642 | 0.01628 | 0/0 | 7.00 |

The batch64 speedup is not a headline: the minimum speedup versus scalar
resident lookup is `0.9838x`, effectively parity under the current Python/NumPy
implementation. The useful systems result is exact batched parity, sub-0.25 ms
resident decode, and precise 7-byte packet-record amortization.

## Interpretation

Promote:

- This closes the strongest Mac-local systems gate after the prompt/logprob PQ
  receiver failed. The deterministic PQ receiver is a small resident lookup
  kernel with exact batch-size-invariant predictions.
- The result covers all six n500 geometry variants, including the OPQ and
  protected-Hadamard mitigations that address lookup-risk concerns.
- The systems contribution is now more than static byte counting: it has a
  measured receiver microkernel, packet-record accounting, exact ID/prediction
  hashes, and mismatch checks.

Do not overclaim:

- This is not a production vLLM/GPU serving result.
- This is not a TTFT, TPOT, goodput, HBM-bandwidth, or interconnect
  measurement.
- This is not protocol-free latent reasoning; it is a source-private PQ packet
  with public decoder side information.
- OPQ, PQ, Hadamard rotations, and table lookup are established primitives; the
  contribution is their use and evaluation as source-private residual packets
  under destructive controls and boundary accounting.

## Readiness Impact

The current technical contribution set is now cleaner:

1. strict source-private benchmark/control protocol,
2. frozen target-verifier packet consumption,
3. geometry-mitigated product-codebook residual packets with source-causal
   n500 lift,
4. byte-causality and lookup-risk diagnostics,
5. receiver-kernel systems evidence with exact batch parity and packet-record
   amortization.

Comfortable ICLR still needs at least one of:

- broader non-hand-decoded receiver evidence,
- native GPU/vLLM/KV TTFT/TPOT/goodput telemetry,
- larger frozen verifier scale,
- or a less synthetic/cross-family benchmark where the method remains positive
  under the same destructive controls.

COLM workshop status is much stronger: this is now a coherent scoped method
paper about source-private evidence packets, not just an evaluation benchmark.

## Next Exact Gate

Mac-local: extend the packet-ring transport microbench to 7-byte PQ records and
14-byte query-aware text records, then join it with this receiver-kernel gate in
one systems waterfall.

GPU-later: reproduce the receiver path inside a serving loop with NVIDIA
counters, reporting TTFT, TPOT, goodput, HBM bytes, PCIe/NVLink traffic, and
selected KV/cache exposure against C2C/KVComm-style baselines.

## Tests

```bash
./venv_arm64/bin/python -m py_compile scripts/build_source_private_pq_receiver_batch_microbench.py
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_pq_receiver_batch_microbench.py -q
```

Outcome: `2 passed in 0.20s`.
