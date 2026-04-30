# Balanced Cross-Family and Model-Receiver References

- date: `2026-04-30`
- purpose: grounding for the cross-family public-only falsification and the
  balanced frozen-target receiver probe.

## Communicating Activations Between Language Model Agents

- source: https://arxiv.org/abs/2501.14082
- blocker helped: broad cross-model communication and activation handoff are
  already active adjacent work.
- mechanism idea: source/target agents can exchange internal activations at
  intermediate layers, which is a much higher-rate interface than a 2-byte
  diagnostic packet.
- next experiment change: keep activation handoff as a high-rate baseline if
  the latent branch reopens; do not claim broad activation communication from
  the current packet gate.
- role: baseline / claim boundary.

## Cache-to-Cache Direct Semantic Communication Between LLMs

- source: https://arxiv.org/abs/2510.03215
- blocker helped: closest cache-level competitor for cross-LLM communication.
- mechanism idea: transfer projected/fused source KV/cache state into the
  target-side computation.
- next experiment change: any future cross-family latent or cache result must
  compare against C2C-style assumptions; the current packet lane should compare
  boundary traffic and access assumptions, not claim direct C2C dominance.
- role: baseline / systems boundary.

## KVComm / Selective KV Sharing

- source: https://arxiv.org/abs/2510.03346
- blocker helped: full-KV byte accounting is not the strongest systems
  baseline if selective KV sharing is available.
- mechanism idea: select and transmit informative KV layers/tokens rather than
  the entire cache.
- next experiment change: add selective-KV byte floors to the systems frontier
  before making broad systems superiority claims.
- role: baseline / ablation.

## TurboQuant

- source: https://arxiv.org/abs/2504.19874
- blocker helped: low-bit KV quantization narrows easy byte-count arguments.
- mechanism idea: rotate, quantize, and use residual correction for compressed
  KV/state transport.
- next experiment change: use TurboQuant-style compressed-state floors when
  comparing against methods allowed to expose source KV/cache state.
- role: systems/compression baseline.

## KIVI

- source: https://arxiv.org/abs/2402.02750
- blocker helped: establishes strong 2-bit KV-cache compression as a standard
  competitor.
- mechanism idea: asymmetric key/value quantization with different granularity
  for keys and values.
- next experiment change: keep KIVI in the systems comparison table for
  same-model and cache-access settings.
- role: baseline.

## Wyner-Ziv Source Coding

- source: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
- blocker helped: explains why tiny source-private packets can be useful when
  the receiver already has correlated public side information.
- mechanism idea: encode only the residual information needed by a decoder with
  side information.
- next experiment change: frame the balanced diagnostic result as
  source-private side-information coding, not protocol-free latent transfer.
- role: theory support.

## Distributed Indirect Source Coding With Decoder Side Information

- source: https://arxiv.org/abs/2405.13483
- blocker helped: formalizes tasks where the decoder recovers a latent task
  variable rather than reconstructing the source.
- mechanism idea: source messages should be evaluated by downstream task
  distortion under decoder side information.
- next experiment change: report packet accuracy and task lift as a
  rate-distortion curve over byte budgets.
- role: theory / framing.

## D3ToM / Diffusion-Transformer Token Merging

- source: https://arxiv.org/abs/2511.12280
- blocker helped: diffusion-transformer work suggests adaptive token/packet
  routing, but not an immediate cross-LLM baseline.
- mechanism idea: use decoder-state or previous-step information to decide
  which tokens are worth transmitting or merging.
- next experiment change: later try decoder-conditioned adaptive packet budgets
  after the fixed-rate receiver is stable.
- role: inspiration / ablation.

## DistServe

- source: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- blocker helped: systems reviewers need TTFT/TPOT/goodput/SLO evidence, not
  only bytes.
- mechanism idea: separate prefill and decode, then report serving latency and
  throughput under explicit SLOs.
- next experiment change: keep Mac CPU/MPS rows as accounting evidence only;
  native server/GPU telemetry is still required for production systems claims.
- role: systems framing.
