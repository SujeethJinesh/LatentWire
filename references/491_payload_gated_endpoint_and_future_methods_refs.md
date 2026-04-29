# Payload-Gated Endpoint Receiver and Future Method References

- date: `2026-04-29`
- blocker helped: payload-gated parser risk, endpoint valid-output robustness,
  systems novelty, and next non-shallow technical contributions.
- experiment change: scale `label_strict` before promoting endpoint receiver
  robustness; keep audit rows as near-miss/failure after payload-gated
  rescoring; prioritize functional relative stitchers, denoising WZ packets,
  and TurboQJL innovation packets as future method branches.

## Systems and Baseline Positioning

1. [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
   - blocker helped: prevents overclaiming local endpoint proxy as serving
     throughput.
   - design idea: future GPU gate should report serving metrics under a real
     scheduler rather than only local generation timing.
   - experiment change: use server-side TTFT/throughput once NVIDIA GPUs are
     available.
   - role: systems baseline.

2. [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics.html)
   - blocker helped: standardizes TTFT, inter-token latency, E2E latency, and
     token accounting.
   - design idea: align LatentWire endpoint logs with common serving metrics.
   - experiment change: future manifests should include TTFT/TPOT-style fields
     where the endpoint supports them.
   - role: measurement standard.

3. [DistServe](https://arxiv.org/abs/2401.09670)
   - blocker helped: distinguishes prefill/decode SLOs from raw local timing.
   - design idea: packet relays should be compared under prefill-heavy and
     decode-heavy regimes.
   - experiment change: server gate should report SLO-aware goodput.
   - role: serving baseline.

4. [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/) and
   [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/)
   - blocker helped: prompt-compression baselines could weaken a byte-savings
     claim.
   - design idea: compare against learned/extractive visible-text compression,
     not only hand-written structured text.
   - experiment change: add a rate curve with compressor cost if learned prompt
     compression is used.
   - role: prompt-compression baseline.

5. [SnapKV](https://arxiv.org/abs/2404.14469),
   [CacheBlend](https://arxiv.org/abs/2405.16444), and
   [LMCache](https://docs.lmcache.ai/)
   - blocker helped: KV/cache reuse can dominate systems value when full
     evidence is reusable.
   - design idea: frame LatentWire as extreme-rate endpoint handoff, not a
     replacement for cache transport.
   - experiment change: compare packet relay against cached full-evidence relay
     when GPU serving exists.
   - role: KV/cache systems baselines.

6. [Toolformer](https://arxiv.org/abs/2302.04761) and
   [ReAct](https://arxiv.org/abs/2210.03629)
   - blocker helped: structured tool traces are a natural high-byte baseline.
   - design idea: packet is a compact tool-result handoff code.
   - experiment change: keep full trace/log relay as an oracle and structured
     trace relay as a high-byte baseline.
   - role: agent/tool-use baseline and framing.

## Future Method Inspiration

7. [Diffusion Transformers](https://arxiv.org/abs/2212.09748),
   [Diffusion of Thoughts](https://arxiv.org/abs/2402.07754), and
   [Block Diffusion](https://arxiv.org/abs/2503.09573)
   - blocker helped: current static packets are shallow and cross-family
     asymmetric.
   - design idea: denoise a corrupted target-side candidate posterior using a
     rate-capped source packet.
   - experiment change: implement denoising WZ packets with 1/2/4 refinement
     steps and source-destroying controls.
   - role: method inspiration.

8. [I-JEPA](https://arxiv.org/abs/2301.08243) and
   [LLM-JEPA](https://arxiv.org/abs/2509.14252)
   - blocker helped: learned packet methods need a non-generative objective
     that predicts useful latent evidence without collapse.
   - design idea: train a receiver to predict missing source-private candidate
     evidence from public context plus a compact packet.
   - experiment change: add anti-collapse and target-preservation losses to a
     learned query-bottleneck receiver.
   - role: objective inspiration.

9. [TurboQuant](https://arxiv.org/abs/2504.19874) and
   [QJL](https://arxiv.org/abs/2406.03482)
   - blocker helped: protected residual packets are source-control positive but
     miss latency and high-budget scalar-preservation thresholds.
   - design idea: rotate/project source innovation vectors before low-bit
     transport, then decode with target side information.
   - experiment change: implement TurboQJL innovation packets at 2/4/6/8 bytes
     against scalar WZ, QJL, raw sign, and protected residual controls.
   - role: quantized transport baseline and inspiration.

10. [Relative Representations](https://openreview.net/forum?id=SrC-nwieGJ) and
    [Model Stitching](https://arxiv.org/abs/2303.11277)
    - blocker helped: static AR-SIP fails bidirectionally across family.
    - design idea: learn a target-preserving relative stitcher over anchor
      features instead of shipping static sparse coordinates.
    - experiment change: train a functional relative stitcher with hard
      2/4/6-byte packetization and anchor-permutation controls.
    - role: cross-family method inspiration.

## Current Gate Outcome

Payload-gated rescoring fixed a parser loophole: diagnostic-code mapping now
requires the generated code to have been transmitted in the source payload.
This demotes audit endpoint rows to near-miss/failure under the
`valid_prediction_rate >= 0.95` gate, even though source accuracy remains high.
Core `n=64` audit reaches packet `0.750`, target `0.250`, best
source-destroying control `0.203`, and full-log p50 TTFT `+260.2 ms`, but
packet valid rate is `0.781`.

The new `label_strict` receiver prompt passes `n=16`, `n=32`, and `n=64` on
core and holdout with exact-label outputs and valid rate `1.000`. At `n=64`,
core reaches packet `0.703` versus target/control `0.250`, and holdout reaches
packet `0.672` versus target/control `0.250`. It is now the live endpoint
branch.

Paired uncertainty on the `n=64` label-strict rows now passes with `5000`
bootstrap samples. The minimum packet-vs-target and packet-vs-best-source-
destroying-control lower CIs are both `+0.297`; the minimum strict-label
packet-vs-target lower CI is `+0.281`; packet valid rate is `1.000`. Query-aware
diagnostic text is accuracy-comparable but costs `14` bytes versus the `2` byte
packet, so it should be framed as a rate/quality comparator. This changes the
next experiment to frozen `n=160` label-strict core+holdout before endpoint
receiver robustness is promoted further.

Core `n=160` now passes, including paired uncertainty. Packet accuracy is
`0.675` and strict-label accuracy is `0.662` versus target/matched-byte text
`0.250`, best source-destroying control `0.250`, and valid rate `1.000`.
Paired lower CIs are `+0.350` versus target and best control. This partially
clears the scale-up rung, but holdout `n=160` remains the exact blocker before
the endpoint receiver branch can be called medium-confirmed.
