# Native Readiness Systems References, 2026-05-01

## Role

This memo supports the source-private native-readiness ledger. Its purpose is
to keep the systems claim honest: current LatentWire evidence is Mac-local
packet accuracy and transport/accounting evidence, while C2C/KVComm/TurboQuant
and vLLM remain native systems baselines to run.

## Primary Sources

- C2C cache-to-cache communication:
  https://arxiv.org/abs/2510.03215
- KVComm:
  https://openreview.net/forum?id=F7rUng23nw
- KVCOMM:
  https://arxiv.org/abs/2510.03346
- TurboQuant:
  https://arxiv.org/abs/2504.19874
- QJL:
  https://arxiv.org/abs/2406.03482
- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180

## Boundary For This Paper

Safe framing: LatentWire currently demonstrates a source-private packet
interface with small byte budgets and Mac-local packet transport/accounting
evidence. Native systems baselines are listed as pending rows with exact
measurement columns.

Unsafe framing: claiming native throughput, HBM, TPOT/goodput, or superiority
over C2C, KVComm, TurboQuant, QJL, or vLLM before running those methods on a
native NVIDIA/vLLM or SGLang stack.

The readiness ledger should be included in an ICLR appendix or systems table
because it makes the non-claim explicit while still giving reviewers a concrete
path to the full systems experiment.
