# ARC Consistency Repair References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full remains
  blocked.
- Current story: fixed-byte source-private packets are only novel if they
  beat target-public, same-byte text, and source-destroyed controls without
  exposing source KV/cache/state.
- Exact gap: consistency-style repair is a useful inspiration, but the
  Mac-local ARC repair receiver does not yet provide positive evidence.

## Primary Sources

1. Consistency Models. <https://arxiv.org/abs/2303.01469>
   - Boundary: one-step denoising/repair is prior art. LatentWire should claim
     a source-private communication protocol and control surface, not novelty
     in consistency training.

2. Consistency Flow Model Achieves One-step Denoising Error Correction Codes.
   <https://arxiv.org/abs/2512.01389>
   - Boundary: one-step neural decoding of corrupted codewords is close in
     spirit, but it does not address cross-model private evidence, target-side
     side information, or source-destroying controls.

3. Prefix-Tuning. <https://arxiv.org/abs/2101.00190>
   - Boundary: continuous prefixes are established target-conditioning
     parameters, so any soft/prefix receiver needs a zero-byte target-only
     control.

4. The Power of Scale for Parameter-Efficient Prompt Tuning.
   <https://arxiv.org/abs/2104.08691>
   - Boundary: prompt tuning reinforces that target-side learned controls can
     work without source communication.

5. Learning to Compress Prompts with Gist Tokens.
   <https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html>
   - Boundary: prompt compression is prior art; LatentWire must differ through
     per-example source-private packets and destructive source controls.

6. LLMLingua. <https://aclanthology.org/2023.emnlp-main.825/>
   - Boundary: matched-rate visible-text compression is a required comparator,
     not the same as a hidden/source-private packet.

7. In-context Autoencoder. <https://arxiv.org/abs/2307.06945>
   - Boundary: compressing context into memory slots for the same LLM is
     different from cross-model fixed-byte packet communication.

8. Activation Engineering. <https://arxiv.org/abs/2308.10248>
   - Boundary: activation steering is prior art. The LatentWire distinction is
     per-example matched-source necessity, not generic target steering.

9. Cache-to-Cache. <https://arxiv.org/abs/2510.03215>
   - Boundary: C2C projects/fuses source KV cache, so it is a high-rate
     exposed-state competitor against source-private packets.

10. KVComm. <https://arxiv.org/abs/2510.03346>
    - Boundary: selective KV sharing still transmits internal source state,
      while LatentWire's systems claim is fixed-byte task evidence with no raw
      source KV exposure.

11. KIVI. <https://arxiv.org/abs/2402.02750>
    - Boundary: KV-cache quantization supplies byte-floor comparators, not a
      source-private communication method.

12. KVQuant. <https://arxiv.org/abs/2401.18079>
    - Boundary: low-bit KV serving compression is a systems baseline; it does
      not replace source-destroying communication controls.

## Reviewer Objection

The sharp objection is still: the packet may be a coded label, target-side
prior, or compressed public/KV state rather than source communication.

The exact control answer is a paired same-byte gate where packet bytes,
framing, decoder, target context, candidates, and target-public base are held
fixed while the source is replaced by zero, wrong-row, same-norm noise,
train-mean source, target-derived source, candidate-roll, label-shuffle, and
matched visible text. The method can be claimed only if matched source wins
and all source-destroyed controls stay at the target floor with paired
uncertainty.

## Decision

The consistency repair branch is not paper-positive. It does sharpen the
claim boundary: the paper's novelty is fixed-byte source-private communication
under source-destroying controls, not denoising, prompt compression, activation
steering, or KV-cache transfer.
