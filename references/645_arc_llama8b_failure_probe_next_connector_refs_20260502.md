# References: ARC Llama-8B Failure Probe And Next Connector

Web/literature check: 2026-05-02. Scope:
`results/source_private_arc_llama8b_failure_probe_20260502/`, the
source-choice failure diagnosis, and the next learned connector branch.

## Local Artifact

- Script: `scripts/analyze_source_private_arc_llama8b_failure_probe.py`
- Result: `results/source_private_arc_llama8b_failure_probe_20260502/`
- Pass gate: `False`
- Best overall router: `source_matches_llama_prediction`
- Best overall router deployable without source index: `False`
- Best deployable router: `packet_margin_ge:0.131799`
- Validation/test best-deployable-router accuracy: `0.408333/0.361522`
- Test Llama/Qwen oracle accuracy: `0.532347`
- Test source/Qwen oracle accuracy: `0.613108`
- Test same-byte-text minus Llama packet: `+0.126427`
- Test source-to-Llama-packet loss: `0.185624`

## Diagnosis

The Llama source-choice branch is not dead because the source model lacks
signal. It is dead as a paper method because the current packet codec loses too
much of that source signal and the strongest conditional rule uses an
audit-only source-selected index. Same-byte visible text still beats the
source-private packet, so the branch should move toward a learned packet or
soft-prefix connector rather than more scalar routing.

## Primary Sources And Boundaries

1. BLIP-2 / Querying Transformer
   - https://arxiv.org/abs/2301.12597
   - Boundary: frozen encoders and frozen language models can be bridged with
     a small learned query module. This motivates a LatentWire query bottleneck
     over source hidden/KV states, but BLIP-2 is multimodal and not
     source-private fixed-byte communication.

2. Flamingo / Perceiver resampler and gated cross-attention
   - https://arxiv.org/abs/2204.14198
   - Boundary: fixed latent resampling plus gated injection into a frozen LM is
     a useful architectural prior for source-to-target communication. It is not
     a fixed-byte private packet baseline.

3. Prefix-tuning / virtual tokens
   - https://arxiv.org/abs/2101.00190
   - Boundary: prefix or soft-token injection is the cleanest target-side
     receiver interface for learned packets. LatentWire must distinguish this
     from ordinary task-specific prompt tuning by requiring source-private
     packets and source-destroying controls.

4. Relative representations
   - https://openreview.net/forum?id=SrC-nwieGJ
   - https://arxiv.org/abs/2209.15430
   - Boundary: relative/shared-anchor coordinates motivate a common language
     but are insufficient alone; LatentWire's recent PCA/transport/RFF
     common-basis gates failed on the ARC disagreement surface.

5. C2C direct semantic communication
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://arxiv.org/abs/2510.03215
   - Boundary: C2C is the hard learned-latent competitor. It transfers/fuses KV
     cache state, while LatentWire's claim must remain fixed-byte and
     source-private unless native cache baselines are measured.

6. KVComm / KVCOMM
   - https://openreview.net/forum?id=F7rUng23nw
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Boundary: selective/online KV communication should be measured as
     source-state communication, not as source-private packet transfer.

7. TurboQuant and QJL
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2406.03482
   - Boundary: these are rate-distortion and low-bit source-state baselines.
     They strengthen the need for native byte/latency/HBM rows, but do not
     solve cross-model source-private reasoning.

8. vLLM/PagedAttention and SGLang/RadixAttention
   - https://arxiv.org/abs/2309.06180
   - https://arxiv.org/abs/2312.07104
   - Boundary: native serving comparisons need TTFT, TPOT, goodput, memory,
     HBM/PCIe/NVLink traffic, and paired accuracy. The current Mac-local probe
     is not a native systems win.

## Next Gate

Implement a frozen 16-64 query bottleneck or soft-prefix connector that:

- reads source hidden/KV states or source-side query summaries;
- emits a fixed-size packet or target prefix;
- trains only on train/validation surfaces with source-destroying controls;
- evaluates on frozen ARC disagreement rows first;
- compares against target-only, Qwen-substituted/cached packets, same-byte
  visible text, shuffled/zero source, and C2C/KV-transfer proxy rows.

If that connector cannot beat Qwen-substituted and same-byte/control baselines
under paired uncertainty, do not widen to more benchmarks.
