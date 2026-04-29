# Systems, Novelty, And Future-Method References

- date: `2026-04-29`
- role: primary-source memo for the source-private diagnostic-packet paper
- blocker: reviewers may classify the current method as a narrow codebook task unless we position it precisely against cache/activation communication, prompt compression, tool-agent handoff, semantic communication, and source coding with decoder side information.

## Near Competitors And Baselines

### C2C / Cache-To-Cache

- Source: [C2C: Cache-to-Cache](https://arxiv.org/abs/2510.03215)
- Blocker helped: direct comparison against latent/KV cross-model communication.
- Mechanism/design idea: learned cache projection/fusion is a high-rate internal-state baseline.
- Next experiment change: keep C2C/KV methods as a separate high-rate baseline lane; compare on rate, model access, and controls rather than claiming they solve the same low-rate packet problem.
- Role: baseline and framing.

### KVComm

- Source: [KVComm](https://arxiv.org/abs/2510.03346)
- Blocker helped: systems reviewers will ask why not communicate KV/cache state directly.
- Mechanism/design idea: selective KV sharing is a cache-transport analogue of sending only useful source-side state.
- Next experiment change: add a systems table that separates packet bytes from cache/state bytes and flags model-access requirements.
- Role: baseline and systems comparison.

### Communicating Activations Between Language Model Agents

- Source: [Communicating Activations Between Language Model Agents](https://arxiv.org/abs/2501.14082)
- Blocker helped: novelty against latent/activation transfer.
- Mechanism/design idea: direct activation injection is a continuous communication channel between agents.
- Next experiment change: do not claim broad latent communication from the packet method; use activation exchange as a broader but higher-access baseline family.
- Role: baseline and limitation framing.

### LLMLingua

- Source: [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)
- Blocker helped: prompt-compression baseline.
- Mechanism/design idea: token-level compression of visible context.
- Next experiment change: maintain matched-byte structured text relay and full-log relay rows, because public prompt compression does not by itself test source-private residual transfer.
- Role: text-compression baseline.

### AutoGen

- Source: [AutoGen](https://openreview.net/forum?id=BAakY1hNKS)
- Blocker helped: practical multi-agent comparison.
- Mechanism/design idea: agents exchange natural-language messages and tool results.
- Next experiment change: position diagnostic packets as a low-rate handoff protocol for cases where full text/tool traces are costly or sensitive.
- Role: framing and practical baseline family.

### ReAct

- Source: [ReAct](https://arxiv.org/abs/2210.03629)
- Blocker helped: tool/reasoning trace relay baseline.
- Mechanism/design idea: interleaved reasoning and actions expose tool-use evidence as text.
- Next experiment change: report full hidden-log relay and diagnostic-masked controls to separate packet communication from trace replay.
- Role: baseline and framing.

### Toolformer

- Source: [Toolformer](https://arxiv.org/abs/2302.04761)
- Blocker helped: tool-use prior art.
- Mechanism/design idea: models learn when and how to call tools and consume tool outputs.
- Next experiment change: keep the source-private benchmark focused on cross-agent transfer of evidence after the source has already observed it.
- Role: framing.

### Chain-of-Agents

- Source: [Chain-of-Agents](https://arxiv.org/abs/2406.02818)
- Blocker helped: text handoff in multi-agent long-context systems.
- Mechanism/design idea: worker agents pass natural-language summaries to a manager.
- Next experiment change: structured summaries are fair high-byte baselines, but the main paper claim should be the far-left rate point.
- Role: practical baseline.

## Theory And Systems Framing

### Wyner-Ziv Source Coding

- Source: [Wyner-Ziv source coding with decoder side information](https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf)
- Blocker helped: explain why the source packet can be tiny when the target has public candidate side information.
- Mechanism/design idea: source sends only the residual needed by a decoder that already has side information.
- Next experiment change: describe packet budget, side information, and distortion explicitly; do not claim information-theoretic optimality.
- Role: theory framing.

### Distributed Indirect Source Coding With Decoder Side Information

- Source: [Distributed Indirect Source Coding with Decoder Side Information](https://arxiv.org/abs/2405.13483)
- Blocker helped: formal analogue where a source observes private evidence and a decoder has side information.
- Mechanism/design idea: task distortion under indirect observations and decoder-side information.
- Next experiment change: use this to motivate learned syndrome packets over candidate pools.
- Role: theory support and future-method inspiration.

### TurboQuant

- Source: [TurboQuant](https://arxiv.org/abs/2504.19874)
- Blocker helped: systems-side comparison against aggressive low-bit inference compression.
- Mechanism/design idea: protect important channels/blocks while compressing the rest.
- Next experiment change: for a learned successor, protect the packet/code bits that correspond to high-impact source-private residuals.
- Role: systems inspiration.

### KIVI

- Source: [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- Blocker helped: KV-cache memory and transport baseline.
- Mechanism/design idea: asymmetric K/V quantization preserves important cache structure at low precision.
- Next experiment change: compare packet transport to the byte footprint of a quantized KV slice when cache communication is feasible.
- Role: systems baseline and inspiration.

### QJL

- Source: [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482)
- Blocker helped: randomized low-rate projection baseline.
- Mechanism/design idea: Johnson-Lindenstrauss style sketches preserve similarity under severe quantization.
- Next experiment change: test a random same-byte sketch control before promoting any learned packet.
- Role: systems baseline and ablation inspiration.

## Future Technical Directions

### JEPA / V-JEPA

- Source: [V-JEPA](https://arxiv.org/abs/2404.08471)
- Blocker helped: current packet is hand-designed rather than learned.
- Mechanism/design idea: predict missing/abstract target-side state from context without reconstructing every token.
- Next experiment change: train a candidate-state predictor that encodes only source-private innovations needed by the target decoder.
- Role: future-method inspiration.

### BLIP-2 / Q-Former

- Source: [BLIP-2](https://arxiv.org/abs/2301.12597)
- Blocker helped: narrow bottleneck connector design.
- Mechanism/design idea: a small query transformer extracts a compact set of source features for a frozen target.
- Next experiment change: implement a zero-init gated query bottleneck over source evidence as the next learned interface after syndrome packets.
- Role: future-method inspiration.

### Flamingo / Perceiver Resampler

- Source: [Flamingo](https://arxiv.org/abs/2204.14198)
- Blocker helped: target-preserving cross-attention gate design.
- Mechanism/design idea: frozen LM receives external state through gated cross-attention.
- Next experiment change: use target-preserving gates and source-destroying controls for any latent connector claim.
- Role: future-method inspiration.

### Continuous Diffusion Language Modeling

- Source: [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)
- Blocker helped: iterative latent refinement beyond one-shot packets.
- Mechanism/design idea: denoise/refine continuous representations in latent space.
- Next experiment change: low priority until the packet claim is paper-ready; possible future gate is a target-cache denoiser that receives a quantized source residual.
- Role: future-method inspiration.

## Related-Work Table Contract

The paper should include a compact table with columns:

| Work family | Medium | Strict source-private evidence? | Explicit rate cap? | Needs model internals? | Source-destroying controls? | Our relation |
|---|---|---:|---:|---:|---:|---|
| C2C / KVComm | KV cache | usually no | no or high-rate | yes | not central | high-rate cache baseline |
| Activation communication | activations | not central | no | yes | not central | latent baseline |
| Prompt compression | text tokens | no | yes | no | no | visible-context baseline |
| Tool-agent handoff | text/tool traces | sometimes | no | no | rarely | practical high-byte baseline |
| Semantic communication | latent/vector | varies | varies | often yes | varies | inspiration |
| Source coding with side information | abstract code | yes | yes | no | by design | theory framing |
| LatentWire packets | 2-byte diagnostic packet | yes | yes | no for source emission | yes | method |
