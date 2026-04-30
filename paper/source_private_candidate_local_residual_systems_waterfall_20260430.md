# Candidate-Local Residual Systems Waterfall

- date: `2026-04-30`
- status: branch-specific Mac systems trace for the live candidate-local residual
  receiver
- code: `scripts/build_source_private_candidate_local_residual_systems_waterfall.py`
- artifact:
  `results/source_private_candidate_local_residual_systems_waterfall_20260430/`
- references: `references/548_candidate_local_residual_systems_refs_20260430.md`

## What Changed

The older systems waterfall measured the conditional PQ packet branch. This new
artifact is tied to the current live candidate-local residual receiver, so it
separates:

1. n512 method rows and destructive controls;
2. packet payload/record/cache-line/DMA accounting;
3. current Python nonresident decode latency;
4. receiver-local cold public candidate feature build;
5. resident sparse decode over cached public candidate residuals;
6. source text/KV exposure.

## Main Evidence

Summary artifact:
`results/source_private_candidate_local_residual_systems_waterfall_20260430/candidate_local_residual_systems_waterfall.md`

Headline results:

- pass gate: `true`;
- n512 packet rows passing: `9/9` across seeds `47/53/59` and directions
  `core_to_holdout`, `holdout_to_core`, and `same_family_all`;
- 8B packet payload is represented as an 11B transport record under the current
  3B header/parity accounting;
- single-request packet traffic rounds to one 64B cache line and one 128B DMA
  burst;
- batch-64 amortized traffic is `11.00` 64B-line bytes/request and `12.00`
  128B-DMA bytes/request;
- current Python nonresident packet decode max p50 is `0.303916 ms/request`;
- representative seed59 n512 resident sparse decode over cached public
  candidate residuals has max p50 `5.231934 us/request`, max p95
  `13.923666 us/request`, and `0` mismatches versus the Python decoder;
- max cold public candidate feature build is `4.161215 ms/request`;
- source text exposed: `false`;
- source KV exposed: `false`;
- calibration/eval family-qualified exact-ID overlap max: `0`;
- transformed held-out eval surface overlap max: `0`.

## Interpretation

This strengthens the systems side of the live method without pretending to be a
production serving result. The online source communication is a byte-scale
packet; the larger public receiver cache is explicitly receiver-local state.
Cold MiniLM/candidate feature construction is not free and is reported
separately from the resident sparse decode path.

Layman explanation: after the receiver builds its public cheat sheet for the
candidate answers, the source only sends a tiny private clue. The expensive part
is preparing the receiver's public cheat sheet; using the clue once that sheet
is ready is very small and fast on the Mac microbench.

## What This Lets Us Claim

Safe claim:

> On the n512 held-out gate, the live candidate-local residual receiver is a
> source-private 8B packet method with explicit 11B record accounting, no source
> text/KV exposure, zero eval-disjoint exact-ID overlap, and an exact resident
> sparse-decode Mac trace over cached public candidate residuals.

Unsafe claim:

> This is not yet a vLLM/NVIDIA serving speedup, not HBM/PCIe/NVLink evidence,
> and not a demonstrated win over C2C/KVComm or KV-cache quantization baselines.

## Remaining ICLR Gap

The next systems gate is a matched NVIDIA/vLLM run with TTFT, TPOT/inter-token
latency, goodput/SLO, GPU memory, HBM traffic, and PCIe/NVLink bytes. The next
method gate is a same-slice competitor table: target-only, matched-byte text,
Relative Representations, linear/Procrustes/CCA public calibration, and a
C2C/KVComm-style cache-access upper bound or scoped proxy.

## COLM Workshop Readiness

This is enough for a COLM systems-positioning table if the paper keeps the
claim narrow: far-left-rate source-private packet communication with public
receiver-side side information, not production serving superiority.
