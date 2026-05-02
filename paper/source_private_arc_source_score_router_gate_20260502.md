# ARC Source-Score Source-Family Router Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: the strongest defensible core is fixed-byte source-private
  packet transfer, public-basis ARC/OpenBookQA packet methods, and
  byte/exposure systems accounting with strict negative gates.
- Exact gap: the ARC Fourier/anchor-syndrome packet is still source-family
  fragile. Scalar confidence from the source models does not repair the
  TinyLlama-vs-Qwen disagreement slice, so ICLR still needs a learned
  common-basis connector or stronger alternate source.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_arc_challenge_source_score_router_gate.py \
  --output-dir results/source_private_arc_challenge_source_score_router_gate_20260502 \
  --bootstrap-samples 500
```

Primary artifact:
`results/source_private_arc_challenge_source_score_router_gate_20260502/source_score_router_gate.json`.

## Lay Explanation

TinyLlama and Qwen sometimes pick different answers, which means their tiny
packets also point in different directions. This experiment let each source
model attach a small "how confident am I?" side channel, then trained a tiny
validation-only rule to decide whether to trust the TinyLlama packet or the
Qwen-substituted packet. If this worked, the earlier failure would mostly be a
confidence-calibration problem. It did not work on frozen test rows.

## Result

- selected validation rule: `source_index_pair_lookup`;
- validation router/Qwen: `0.451/0.389`, delta `+0.063`, CI95 low `+0.010`;
- frozen test router/Qwen/oracle: `0.315/0.317/0.586`;
- frozen test router-minus-Qwen mean/min: `-0.002/-0.002`;
- frozen test CI95 low versus Qwen: `-0.031`;
- source-score sidecar: `1B`;
- disagreement rows: `144` validation, `473` test;
- score-cache audit: TinyLlama and Qwen score caches match the parent
  source-choice caches and contain no forbidden answer fields.

The best scalar source-confidence row also fails to promote. The
`negative_qwen_neg_entropy` rule ties Qwen-substituted packets on test at
`0.317`, with CI95 low `-0.018`; `alt_margin_minus_qwen_margin` hurts test
accuracy at `0.309`.

## Decision

Rule out scalar source-confidence routing for this ARC source-family repair
surface. The validation improvement was real on the validation rows but did
not transfer to frozen ARC test disagreement rows. The next positive branch
should be one of:

1. a learned common-basis connector trained without answer leakage;
2. a stronger alternate source on NVIDIA hardware;
3. a continuous/structured source packet that is not reducible to selected
   candidate index plus scalar confidence.

Do not claim that source-side confidence solves cross-model latent
communication. Use this gate as reviewer-facing evidence that we tried and
ruled out the obvious low-cost repair.

## Positioning

This is closest to confidence-based routing and selective prediction, but it
is a stricter source-family packet repair test because the selector must choose
between two source-private packet senders on rows where the senders disagree.
It also sharpens the boundary against prefix tuning and KV/cache transfer: we
are not optimizing a token prefix or moving source KV state; the tested object
is a fixed packet plus a one-byte source-risk sidecar.

Relevant prior work to cite when describing the negative gate and next method:
RACER confidence-routing (`https://arxiv.org/abs/2603.06616`),
confidence-driven LLM routing (`https://arxiv.org/abs/2502.11021`), semantic
uncertainty (`https://arxiv.org/abs/2302.09664`), semantic entropy
(`https://www.nature.com/articles/s41586-024-07421-0`), relative
representations (`https://arxiv.org/abs/2209.15430`), semantic channel
equalization (`https://arxiv.org/abs/2411.19719`), SAE features
(`https://arxiv.org/abs/2309.08600`), SAE universality
(`https://arxiv.org/abs/2410.06981`), prefix tuning
(`https://arxiv.org/abs/2101.00190`), C2C
(`https://arxiv.org/abs/2510.03215`), KVComm
(`https://arxiv.org/abs/2510.03346`), KVCOMM
(`https://arxiv.org/abs/2510.12872`), and TurboQuant
(`https://arxiv.org/abs/2504.19874`).
