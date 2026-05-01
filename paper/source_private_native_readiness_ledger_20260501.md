# Source-Private Native Readiness Ledger, 2026-05-01

## Status

- paper readiness: not ICLR-ready; this is a systems boundary artifact, not a
  native systems win.
- current story: LatentWire has Mac-local source-private packet accuracy,
  byte-boundary, and packed-record transport evidence.
- exact blocker: native NVIDIA/vLLM or SGLang measurements are still required
  before claiming production throughput, HBM traffic, or wins over native
  cache/KV/quantization systems.

## Artifact

- code: `scripts/build_source_private_native_readiness_ledger.py`
- output: `results/source_private_native_readiness_ledger_20260501/`
- references: `references/566_native_readiness_systems_refs_20260501.md`

Lay explanation: this is a checklist separating what we measured on the Mac
from what we still need to measure on real serving hardware. It prevents the
paper from accidentally saying "we beat GPU KV-cache systems" when the current
evidence only supports "our packet is tiny, source-private, and works in the
local benchmark."

## Readout

The ledger has `3` local measured rows:

- train-donor anti-shuffle packet frontier: source-private, cross-model,
  selected n512 cross-family accuracy minimum `0.652`, best selected control
  maximum `0.273`, `12-14B` frontier.
- packet-ring packed-record microbench: source-private packed packet transport,
  batch-64 p50 `0.669 ns/request`, `5` line bytes/request for the measured
  5B record contract.
- Mac endpoint packet proxy: source-private endpoint proxy, minimum endpoint
  packet accuracy `0.675`, TTFT proxy `471.636 ms`.

It also has `5` pending native rows: C2C, KVComm/KVCOMM, TurboQuant-style
low-bit KV, QJL-style sign sketches, and vLLM/PagedAttention.

## Interpretation

Safe claim: Mac-local source-private packet accuracy, byte-boundary, and
transport proxy evidence.

Unsafe claim: native GPU throughput, HBM traffic, production serving goodput,
or beating C2C/KVComm/TurboQuant/QJL/vLLM in their native systems setting.

The ICLR systems contribution should therefore be written as an interface and
measurement-boundary contribution now, with a clear next-native-gate table. If
NVIDIA hardware lands, the ledger already defines the rows to fill: TTFT, TPOT,
throughput, peak memory, HBM read/write bytes, source exposure, and matched
accuracy for packet/text/KV baselines.

## Next Gate

Port the selected packet receiver into a native serving harness and run
packet, visible-text, source-KV, C2C/KVComm, and low-bit KV baselines under the
same source-private task rows. Until then, the systems section should not claim
native production superiority.
