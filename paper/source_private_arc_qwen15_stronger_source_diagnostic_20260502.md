# ARC Qwen-1.5B Stronger-Source Diagnostic

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is stronger after this diagnostic,
  but ICLR full paper remains blocked.
- Current story: fixed-byte ARC common-basis packets can transmit useful
  source decisions when the alternate source is stronger, but the evidence is
  still same-family and validation-gate-incomplete.
- Exact gap: this is not the required cross-family proof. We still need a
  strict cross-family pass or a learned hidden/query common-basis connector
  that survives validation selection, frozen test, seed repeats, and
  destructive controls.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_arc_challenge_source_family_cache_falsification.py \
  --output-dir results/source_private_arc_challenge_source_family_cache_falsification_20260502_qwen15_cpu \
  --alternate-source-family qwen2.5_1.5b \
  --alternate-source-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --source-lm-device auto_cpu \
  --source-lm-dtype float32 \
  --source-lm-max-length 256 \
  --source-lm-normalization mean \
  --source-lm-prompt-mode qa \
  --bootstrap-samples 500
```

Primary artifact:
`results/source_private_arc_challenge_source_family_cache_falsification_20260502_qwen15_cpu/source_family_cache_falsification.json`.

## Lay Explanation

TinyLlama was a weak alternate source, so its packets failed on the hardest
rows. This run asks a stronger local source model, Qwen-1.5B, to choose the
answers, then sends the same 12-byte public-basis packet to the receiver. On
test examples where Qwen-1.5B and the original Qwen-0.5B source disagree, the
Qwen-1.5B packet wins strongly. That says the packet can carry useful stronger
source decisions. It does not yet prove cross-family latent communication,
because both source models are Qwen-family models and the validation
disagreement slice did not clear the strict paired-uncertainty gate.

## Result

- overall pass gate: `False`;
- alternate source family: `qwen2.5_1.5b`;
- validation full-slice pass: `5/5` seeds;
- validation Qwen-disagreement pass: `0/5` seeds because CI95 low versus
  Qwen-substituted packets is `-0.021` on only `95` disagreement rows;
- frozen test full-slice pass: `5/5` seeds;
- frozen test full-slice matched/target/text: `0.442/0.265/0.401`;
- frozen test Qwen-disagreement pass: `5/5` seeds on `388` rows;
- frozen test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.482/0.184/0.456/0.296`;
- frozen test minimum matched-minus-Qwen-substituted: `+0.294`;
- frozen test CI95 low versus Qwen-substituted: `+0.216`.

## Decision

Promote this as a promising same-family stronger-source diagnostic, not as an
ICLR-ready cross-family claim. It weakens the pessimistic interpretation of the
TinyLlama failure: the ARC packet mechanism is not merely copying the original
Qwen-0.5B source cache, and it can carry decisions from a stronger source.

The next exact gate should keep this source-strength direction but make it
reviewer-clean:

1. run a true cross-family stronger source on NVIDIA, using the same frozen
   ARC disagreement protocol;
2. repeat Qwen-1.5B with a larger validation/control slice if more Mac time is
   available;
3. compare the stronger-source packet against native cache-transfer baselines
   with byte/exposure accounting once NVIDIA systems access lands.

## Positioning

This diagnostic is still much lower-rate than C2C/KVComm/KVCOMM because it
ships a fixed 12-byte packet rather than source KV/cache state. The safe claim
is stronger-source packet transfer under a public-basis packet contract, not
general model-to-model latent reasoning.
