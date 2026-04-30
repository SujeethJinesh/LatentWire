# Masked Consistency Receiver References

- date: `2026-04-30`
- purpose: primary-source grounding for the learned one-step masked-consistency
  receiver over source-private learned syndrome packets.

## D3PM

- source: https://arxiv.org/abs/2107.03006
- blocker helped: learned packet receivers operate over discrete bytes/bits,
  not Gaussian image latents.
- mechanism idea: use structured corruptions such as mask, random byte,
  shuffled source, wrong projection, and target-derived packets.
- next experiment change: train the receiver on clean and corrupted packet
  views, with corruptions forced back to target-prior behavior.
- role: inspiration / ablation design.

## MaskGIT

- source: https://arxiv.org/abs/2202.04200
- blocker helped: the receiver should not need autoregressive private-log
  reconstruction.
- mechanism idea: use bidirectional masked-token style evidence and confidence
  over a small candidate set.
- next experiment change: keep a one-step candidate scorer as the main row and
  treat any iterative refinement as future work.
- role: inspiration.

## Consistency Models

- source: https://proceedings.mlr.press/v202/song23a.html
- blocker helped: reviewers may ask why a one-step noisy-packet receiver is a
  principled learned decoder rather than a heuristic.
- mechanism idea: map noisy/corrupted states directly to the same clean
  endpoint prediction.
- next experiment change: train clean and masked packet views toward the same
  gold candidate while destructive controls map to target prior.
- role: theory support / method framing.

## Latent Consistency Models

- source: https://arxiv.org/abs/2310.04378
- blocker helped: receiver learning should happen in compact latent/candidate
  state, not through private text reconstruction.
- mechanism idea: denoise directly in a low-dimensional latent decision space.
- next experiment change: use candidate posterior scores as the endpoint, not
  generated source text.
- role: inspiration / framing.

## I-JEPA

- source: https://arxiv.org/abs/2301.08243
- blocker helped: the method needs a non-reconstructive latent prediction
  story.
- mechanism idea: predict target-side representations from context without
  reconstructing raw inputs.
- next experiment change: receiver predicts candidate scores from packet and
  public candidate side information, never private log text.
- role: framing / theory support.

## V-JEPA

- source: https://arxiv.org/abs/2404.08471
- blocker helped: avoid overstating text/trace reconstruction as the source of
  communication.
- mechanism idea: feature prediction rather than direct reconstruction.
- next experiment change: preserve the anti-reconstruction stance in the paper:
  the packet is evaluated only by downstream candidate recovery and controls.
- role: framing.

## Wyner-Ziv

- source: https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- blocker helped: source-private packet utility needs a formal side-information
  explanation.
- mechanism idea: the target has public candidate/prior side information, so
  the source only sends conditional residual evidence.
- next experiment change: report receiver inputs as packet plus public
  candidate side information, and keep target-only/target-derived controls.
- role: theory support.

## DISCUS / Syndrome Coding

- source: https://doi.org/10.1109/TIT.2002.808103
- blocker helped: the term "syndrome" should be backed by side-information
  decoding, not used as a metaphor.
- mechanism idea: transmit a compact index that the decoder resolves with
  correlated side information.
- next experiment change: compare learned consistency decoding against the
  deterministic Hamming syndrome decoder and require destructive controls.
- role: theory support / baseline framing.

## Bottom Line

The learned receiver should be framed as a one-step side-information decoder:
clean and masked packets map to the gold candidate; destroyed packets map back
to the target prior. It reduces the hand-written decoder objection only if it
preserves packet utility while suppressing control leakage. It does not yet
prove protocol-free semantic latent transfer.
