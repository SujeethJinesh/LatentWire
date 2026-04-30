# Masked-Consistency Pruning and Next Systems/Receiver References

- date: `2026-04-30`
- role: primary-source memo for pruning the current masked-consistency receiver
  and choosing the next ICLR-strengthening branch

## Sources and Experiment Implications

1. **Consistency Models**
   Source: https://arxiv.org/abs/2303.01469
   Blocker: the current learned receiver is one-step but does not map source
   evidence to the correct packet on balanced `diag_only`.
   Design idea: train a receiver to directly denoise corrupted candidate-packet
   compatibility scores, not only regress a source vector then hash it.
   Next experiment change: only pursue if paired with source-destroying
   negatives and public-only separation.
   Role: inspiration / ablation.

2. **Flow Matching for Generative Modeling**
   Source: https://arxiv.org/abs/2210.02747
   Blocker: ridge packet projection is not enough to move from private evidence
   to target-side candidate state.
   Design idea: learn a small conditional transport from target prior/candidate
   state to source-conditioned posterior over candidates.
   Next experiment change: replace masked-consistency ridge with a posterior
   flow over candidate logits before touching LLM weights.
   Role: inspiration / mechanism.

3. **Diffusion Transformers (DiT)**
   Source: https://arxiv.org/abs/2212.09748
   Blocker: we need a less hand-coded learned receiver but cannot afford a large
   generative model on Mac.
   Design idea: use a tiny transformer/query bottleneck over packet-bit tokens
   and candidate feature tokens as a bounded DiT-style denoiser.
   Next experiment change: only a micro n64 prototype; do not scale unless it
   beats the current ridge receiver under controls.
   Role: inspiration.

4. **I-JEPA**
   Source: https://arxiv.org/abs/2301.08243
   Blocker: current packet learning may overfit surface features instead of
   predicting missing target-side information.
   Design idea: predict masked candidate-state summaries from public context
   plus source packet, with collapse controls.
   Next experiment change: frame receiver loss as missing side-information
   prediction, not answer-label classification alone.
   Role: inspiration / framing.

5. **V-JEPA 2**
   Source: https://arxiv.org/abs/2506.09985
   Blocker: learned receiver needs a stronger anti-collapse objective.
   Design idea: joint embedding prediction with explicit invariance and
   negative/source-destroying examples.
   Next experiment change: add shuffled-source and public-only negatives as
   first-class training views.
   Role: inspiration.

6. **TurboQuant**
   Source: https://arxiv.org/abs/2504.19874
   Blocker: the paper needs a stronger systems contribution than algorithmic
   accuracy alone.
   Design idea: protected/mixed-precision residual packets and rate-distortion
   frontiers are reviewer-legible systems baselines.
   Next experiment change: packet trace-card v2 should report raw bytes,
   cache-line/DMA quanta, exposure, encode/decode latency, and matched-byte
   comparator rows.
   Role: baseline / systems framing.

7. **QJL / Quantized Johnson-Lindenstrauss Compression**
   Source: https://arxiv.org/abs/2406.03482
   Blocker: reviewers may ask whether our packet is just another compression
   baseline.
   Design idea: compare source-private packets against randomized projection
   and quantized residual transport under matched-byte budgets.
   Next experiment change: use QJL/TurboQuant language for protected residual
   packet baselines, but keep the claim boundary that we transmit private
   evidence, not full KV state.
   Role: baseline / framing.

8. **KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization**
   Source: https://arxiv.org/abs/2402.02750
   Blocker: systems reviewers will compare any cross-model communication
   interface against KV transport/compression.
   Design idea: report KV byte floors and why a 2-byte source-private packet
   occupies a different operating point from compressed KV state.
   Next experiment change: include KIVI/KVQuant rows as memory-traffic baselines
   in the final systems table.
   Role: baseline.

9. **Gist Tokens**
   Source: https://arxiv.org/abs/2304.08467
   Blocker: prompt-compression baselines can undercut packet novelty if ignored.
   Design idea: compare packet bytes against soft/prompt-compressed relays and
   make clear which methods require target-side learned tokens versus explicit
   source-private evidence packets.
   Next experiment change: keep query-aware text and prompt-compression rows in
   the systems comparison.
   Role: baseline / paper framing.

10. **Slepian-Wolf / Wyner-Ziv Side Information Coding**
    Sources: https://ieeexplore.ieee.org/document/1054744 and
    https://ieeexplore.ieee.org/document/1055349
    Blocker: reviewers need a principled explanation for why a tiny packet can
    help only when target-side candidate context is present.
    Design idea: present LatentWire as source-private residual coding with
    decoder side information.
    Next experiment change: keep deranged-table and public-only controls as the
    experimental analog of destroying decoder side information.
    Role: theory support / framing.
