# ARC Phi-3 Cross-Family Source Diagnostic

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte ARC common-basis packets can transmit useful
  Qwen-family source decisions, and the Qwen-1.5B diagnostic shows source
  strength matters, but the same packet does not yet generalize to a non-Qwen
  source on Mac-local Phi-3.
- Exact gap: the strict cross-family source-family gate is negative. We need a
  stronger non-Qwen source on NVIDIA or a hidden/query common-basis connector
  that survives validation, frozen test, seed repeats, paired uncertainty, and
  destructive controls.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_arc_challenge_source_family_cache_falsification.py \
  --output-dir results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu \
  --alternate-source-family phi3_mini_4k \
  --alternate-source-model /Users/sujeethjinesh/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e \
  --alt-validation-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_validation/source_prediction_cache.jsonl \
  --alt-test-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_test/source_prediction_cache.jsonl \
  --source-lm-device auto_cpu \
  --source-lm-dtype float32 \
  --source-lm-max-length 256 \
  --source-lm-normalization mean \
  --source-lm-prompt-mode qa \
  --bootstrap-samples 500
```

Primary artifact:
`results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/source_family_cache_falsification.json`.

## Lay Explanation

This asks a different model family, Phi-3, to privately choose answers for ARC.
We then compress Phi-3's choice into the same 12-byte public-basis packet and
give that packet to the Qwen receiver. The important test is the subset where
Phi-3 and Qwen-0.5B choose different answers. If Phi-3 packets won there, we
could claim the packet is not just a Qwen cache artifact. They did not win.

## Result

- overall pass gate: `False`;
- alternate source family: `phi3_mini_4k`;
- validation full-slice pass: `0/5` seeds;
- validation Qwen-disagreement pass: `0/5` seeds on `223` rows;
- validation Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.238/0.384/0.238/0.229`;
- frozen test full-slice pass: `0/5` seeds;
- frozen test full-slice matched/target/text: `0.244/0.265/0.241`;
- frozen test Qwen-disagreement pass: `0/5` seeds on `833` rows;
- frozen test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.200/0.340/0.209/0.273`;
- frozen test minimum matched-minus-Qwen-substituted: `-0.143`;
- frozen test CI95 low versus Qwen-substituted: `-0.193`;
- Phi source-choice accuracy before packet: validation `0.274`, test `0.246`.

## Decision

Rule out Mac-local Phi-3 as the cross-family source repair for the current ARC
Fourier/anchor-syndrome packet. This is not a failure of the 12-byte packet
protocol in general: Qwen-1.5B shows that stronger source decisions can be
transmitted on frozen ARC test. It is a failure of this available non-Qwen
source under the current answer-key-forbidden prompt and packet contract.

Do not claim source-family-general ARC communication. The next exact gate
should be one of:

1. a stronger non-Qwen source on NVIDIA with the same frozen ARC disagreement
   protocol;
2. a hidden/query connector that learns a richer common basis from source
   hidden states rather than cached candidate-score geometry;
3. a Q-Former/Perceiver-style query bottleneck or SAE/crosscoder-aligned
   dictionary, evaluated with the same destructive controls and byte/exposure
   accounting.

## Positioning

This negative row is useful for reviewer trust. It separates three claims:

- positive: fixed-byte public-basis packets can help on ARC with the original
  Qwen source cache;
- positive diagnostic: a stronger same-family Qwen-1.5B source repairs frozen
  ARC test disagreements;
- negative: the locally available non-Qwen Phi-3 source is too weak on ARC and
  does not provide cross-family evidence.

Systems implication: the method still has an attractive byte/exposure profile
relative to KV/cache-transfer baselines, but ICLR cannot rest on systems
accounting alone until native NVIDIA serving rows and a robust cross-family
method are available.

## 8B b2000 Follow-Up

Follow-up artifacts:

- plain same-protocol Fourier gate:
  `results/source_private_arc_challenge_fourier_anchor_syndrome_cross_family_phi3_gate_20260502_budget8_10seed_b2000/`;
- strict source-family wrapper:
  `results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000/`;
- memo:
  `paper/source_private_arc_phi3_8b_b2000_cross_family_falsification_20260502.md`.

The stricter follow-up uses the promoted `8B` payload / `11B` framed packet,
10 seeds, and 2000 paired bootstrap samples. It remains negative:

- overall pass gate: `False`;
- full validation/test pass: `0/10` and `0/10`;
- Qwen-disagreement validation/test pass: `0/10` and `0/10`;
- test full-slice matched/target/text: `0.244/0.265/0.232`;
- test Qwen-disagreement rows: `833`;
- test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.200/0.340/0.203/0.273`;
- test Qwen-disagreement CI95 low versus Qwen-substituted: `-0.192`.

Decision update: the current ARC packet is now explicitly same-family/source
cache bounded. Do not present it as source-family-general. The next live branch
must change the source endpoint or the representation basis, not merely repeat
the Phi-3 packet-only path.
