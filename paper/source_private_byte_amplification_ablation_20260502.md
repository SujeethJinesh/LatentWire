# Source-Private Byte-Amplification Ablation

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: LatentWire's strongest current contributions are fixed-byte
  source-private packets with destructive controls, public-basis/innovation
  diagnostics, and systems byte/exposure accounting.
- Exact gap: method-side ICLR readiness still needs a positive receiver that
  proves matched-source necessity under strict controls; systems-side native
  GPU rows remain pending.

## What Changed

Added `scripts/build_source_private_byte_amplification_ablation.py`, a
Mac-local systems artifact builder that holds cached packet predictions fixed
and varies only the communicated object:

- framed LatentWire packet;
- the same packet padded to a single `64B` cache line;
- fp16 source-score vector floor;
- fp16 source-hidden vector floor;
- QJL, KIVI, KVQuant, TurboQuant one-token KV byte floors;
- KVComm 30%-layer fp16 floor;
- C2C one-token fp16 KV floor.

The artifact is generated at
`results/source_private_byte_amplification_ablation_20260502/` and writes
JSON, CSV, Markdown, and a manifest.

Lay explanation: this does not run a new model. It takes the predictions we
already have and asks, "If the same decision crossed the system boundary in
different forms, how many bytes and what private source state would cross?"
That separates the packet interface from cache compression.

## Evidence

Artifact headline:

- pass gate: `True`;
- benchmark rows: `4`;
- interface rows: `40`;
- packet framed-byte range: `4-15B`;
- max single-request cache-line amplification: `16.0x`;
- max single-request DMA amplification: `32.0x`;
- minimum QJL/KV floor: `768B`;
- minimum KV floor versus largest framed packet: `51.2x`;
- minimum KV floor versus `64B` cache-line padded packet: `12.0x`;
- fp16 source-score floor: `8B`, only `2.0x` the minimum packet, but not
  source-private.

Anchored positive packet rows:

| Surface | Accuracy | Packet | 64B Padded | QJL 1-bit KV Floor | TurboQuant 3.5-bit KV Floor | C2C fp16 KV Floor |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge test | `0.344` | `15B` | `64B` | `768B` | `2688B` | `12288B` |
| OpenBookQA test | `0.378` | `6B` | `64B` | `768B` | `2688B` | `12288B` |
| HellaSwag validation1024 | `0.461` | `5B` | `64B` | `768B` | `2688B` | `12288B` |
| HellaSwag full compaction | `0.619` | `4B` | `64B` | `768B` | `2688B` | `12288B` |

The HellaSwag rows remain diagnostic/systems-rate rows, not ICLR headline
accuracy rows, because prior ledgers record label-copy and terminal-tail
limitations.

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_build_source_private_byte_amplification_ablation.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/build_source_private_byte_amplification_ablation.py`

## Interpretation

This strengthens the systems contribution without overclaiming native speed.
It shows that even after worst-case single-request `64B` cache-line padding,
the smallest one-token KV/source-state floor is still `12x` larger and crosses
source KV state. At the same time, the `8B` score-vector row preempts an
important reviewer objection: there are byte-small alternatives, but they
expose raw source scores and are therefore a different threat model.

The paper should frame the systems claim as an interface regime separation:
LatentWire packets are constant-byte task evidence; QJL, TurboQuant, KIVI,
KVQuant, KVComm, and C2C compress or transmit state objects whose bytes scale
with KV/cache elements.

## Decision

Promote the byte-amplification ablation to the systems evidence bundle for
COLM and ICLR. Do not claim native throughput, TTFT, TPOT, HBM traffic, or
baseline defeat until NVIDIA vLLM/SGLang/C2C/KVComm rows are measured.

The next exact method gate remains a candidate-alignment-sensitive receiver
that clears candidate-roll/candidate-derangement controls. The next exact
systems gate after this Mac-local artifact is native NVIDIA serving ingestion.
