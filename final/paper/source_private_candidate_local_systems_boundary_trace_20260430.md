# Candidate-Local Systems Boundary Trace

- date: `2026-04-30`
- status: reviewer-facing systems boundary table for the live candidate-local
  residual receiver
- code: `scripts/build_source_private_candidate_local_systems_boundary_trace.py`
- artifact:
  `results/source_private_candidate_local_systems_boundary_trace_20260430/`
- references:
  `references/551_candidate_local_systems_boundary_trace_refs_20260430.md`

## What Changed

This artifact turns the candidate-local residual systems evidence into a
boundary trace-card. It separates:

1. same-slice measured packet and text-control rows;
2. Mac/accounting rows for query-aware text, full hidden-log relay, and KV byte
   floors;
3. pending native systems rows for C2C, KVComm/KVCOMM, Q-KVComm, TurboQuant,
   CacheGen, and vLLM/PagedAttention.
4. deterministic Mac-side KV proxy byte floors for C2C, KVComm/Q-KVComm,
   TurboQuant, and CacheGen derived from the existing Qwen3 endpoint summaries
   and model KV-cache formula.

## Main Evidence

Summary artifact:
`results/source_private_candidate_local_systems_boundary_trace_20260430/candidate_local_systems_boundary_trace.md`

Headline:

- pass gate: `true`;
- live candidate-local residual packet rows: `9/9`;
- live accuracy range: `0.500-0.625`;
- live packet payload/record bytes: `8B/11B`;
- live batch-64 line bytes/request: `11.00`;
- live resident sparse decode p50: `5.231934 us/request`;
- source text exposed: `false`;
- source KV exposed: `false`;
- measured boundary rows: `13`;
- pending native systems rows: `6`;
- KV native proxy byte-floor rows: `6`;
- minimum KV native proxy payload/live record ratio: `4887.3x`;
- C2C fp16 source-KV proxy payload: `344064B`;
- KVComm 30%-layer fp16 source-KV proxy payload: `103219.2B`;
- TurboQuant 3.5-bit source-KV proxy payload: `75264B`;
- ICLR systems complete: `false`;
- COLM systems table ready: `true`.

The pending native rows are not defeated baselines. They are marked as native
systems work requiring source-KV exposure and NVIDIA/vLLM-style instrumentation:
C2C cache fusion, KVComm/KVCOMM selected KV reuse, Q-KVComm adaptive compressed
KV communication, TurboQuant, CacheGen, and vLLM/PagedAttention serving.

The proxy rows make the systems comparison sharper without overclaiming:

- C2C fp16 source-KV lower-bound proxy: `344064B`;
- KVComm 30%-layer fp16 lower-bound proxy: `103219.2B`;
- Q-KVComm 6x compressed lower-bound proxy: `57344B`;
- TurboQuant 3.5-bit lower-bound proxy: `75264B`;
- TurboQuant 2.5-bit aggressive lower-bound proxy: `53760B`;
- CacheGen 4.3x compressed lower-bound proxy: `80014.9B`.

All of these proxy rows expose source KV/cache state and are byte-floor rows,
not native accuracy/latency measurements.

## Interpretation

This gives the systems contribution a precise claim boundary. The live method
communicates through an 8B source-private payload, carried as an 11B record,
with no source text or source KV exposure. The Mac artifact validates resident
sparse decoding over receiver-local public candidate residuals; it does not
measure GPU serving throughput or HBM traffic. The new KV proxy rows show that
even optimistic source-KV/cache byte floors are thousands of live-record bytes
on the local endpoint proxy, while keeping the caveat that they are not native
kernel results.

Layman explanation: this table says what actually crosses the boundary. Our
row sends a tiny private clue. Text rows send private words. KV/cache systems
move internal model memory. Those are different systems contracts, so the paper
should compare them by bytes, exposure, and native measurement status.

## Safe Claim

On the current n512 candidate-local surface, LatentWire has a source-private
8B-payload / 11B-record packet row with explicit 64B/128B/4KB accounting, no
source text/KV exposure, strict controls, and a Mac resident sparse-decode
trace. The Mac proxy table gives deterministic source-KV byte floors for
C2C/KVComm/Q-KVComm/TurboQuant/CacheGen under the local Qwen3 endpoint summaries
and marks all of them as source-KV exposed. C2C/KVComm/TurboQuant-style rows
remain native systems baselines rather than defeated competitors.

## Non-Claims

- This is not production NVIDIA/HBM/vLLM throughput.
- This does not beat C2C, KVComm, Q-KVComm, TurboQuant, CacheGen, KIVI, or
  KVQuant on native cache/KV tasks.
- KV proxy rows are deterministic byte floors only; they do not measure native
  compression quality, HBM traffic, or serving latency.
- Private text relays are not impossible; they are different exposure and rate
  points.
- The current receiver is not protocol-free latent transfer.

## Remaining ICLR Gap

Comfortable ICLR still needs the native systems rows when hardware is
available: TTFT, TPOT, SLO-goodput, GPU memory, HBM traffic, and explicit source
KV bytes for C2C/KVComm/Q-KVComm/TurboQuant/CacheGen-style baselines. The next
method gate should be a materially new contribution, not another minor RR
repair; the best Mac-bounded option is a TurboQuant-style randomized
candidate-local innovation packet or an RR-gated safety/abstention rule.

## COLM Workshop Use

This is usable for COLM as a systems-positioning table if the claim remains:
source-private packet boundary interface with receiver-side public side
information, not general serving superiority.
