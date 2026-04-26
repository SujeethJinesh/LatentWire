# Qwen2.5-Math SVAMP32 Target-CE Generation Gate

- date: `2026-04-26`
- readiness: not ICLR-ready
- live branch tested: token-local source cross-attention prefix connector trained
  with target-side continuation next-token CE
- scale-up rung: strict-small diagnostic with clean-row generation
- git base: `8105d3ee7c7426931d71e9a2dd04a4a440f197b3`

## Start-Of-Cycle Status

1. current ICLR readiness and distance: not ready; at least one stable positive
   method, medium/large seed repeats, strict source controls, and cross-family
   falsification remain missing
2. current paper story: C2C exposes source-assisted headroom, but source
   readout, sidecar, selector, process-repair, and tiny prefix-emitter methods
   are explained by target priors or source-destroying controls
3. exact blocker to submission: no deployable source-derived row beats
   target/text/C2C-relevant baselines while surviving zero-source,
   shuffled-source, target-only, and slots-only controls
4. live branch: true target-side continuation-loss variant of the source
   cross-attention prefix connector
5. highest-priority gate: check whether target-CE training plus generation-time
   evaluation rescues the failed contrastive/logprob prefix branch
6. scale-up rung: strict-small same-family gate on SVAMP32 exact IDs

## Implementation

Updated `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py` to
support:

- `--training-objective target_ce`, minimizing target continuation
  next-token negative log likelihood
- source-control penalties under the same CE objective
- optional heldout greedy generation with the learned prefix
- generation condition filters and clean-ID-only generation to keep the gate
  cheap but interpretable

The run used a 2-prefix, 16-hidden connector with two outer folds and one
epoch. Logprob was scored on all 32 examples; 64-token generation was decoded
on the six clean C2C-headroom IDs.

## Command

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_cross_attention_logprob_probe.py --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl --target-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl --teacher-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false --feature-layers last --prefix-len 2 --hidden-dim 16 --epochs 1 --outer-folds 2 --training-objective target_ce --source-control-contrastive-weight 0.5 --source-control-contrastive-margin 0.05 --source-control-contrastive-control zero_source --source-control-contrastive-control shuffled_source --length-normalize true --run-generation --generation-max-new-tokens 64 --generation-example-id 3e8a5691f5443495 --generation-example-id 1d50b408c8f5cd2c --generation-example-id de1bf4d142544e5b --generation-example-id 47464cc0b064f172 --generation-example-id 6e9745b37ab6fc45 --generation-example-id 575d7e83d84c1e67 --generation-condition matched --generation-condition zero_source --generation-condition shuffled_source --generation-condition target_only_prefix --generation-condition slots_only_prefix --device mps --train-device mps --dtype float32 --output-json results/qwen25math_svamp32_target_ce_generation_gate_20260426/smoke.json --output-md results/qwen25math_svamp32_target_ce_generation_gate_20260426/smoke.md --generation-output-jsonl results/qwen25math_svamp32_target_ce_generation_gate_20260426/generations.jsonl
```

## Result

Status: `source_cross_attention_logprob_fails_gate`.

Heldout logprob:

- clean IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `4/6`
- clean control leaks: `4/6`
- mean matched margin on clean IDs: `0.061688`
- mean best-control margin on clean IDs: `0.256471`
- mean matched-minus-control clean margin: `-0.194783`

64-token generation on the six clean C2C-headroom IDs:

| Condition | Correct | Clean Correct | Numeric Coverage |
|---|---:|---:|---:|
| matched | 1/6 | 1/6 | 6/6 |
| zero-source | 2/6 | 2/6 | 6/6 |
| shuffled-source | 2/6 | 2/6 | 6/6 |
| target-only prefix | 2/6 | 2/6 | 6/6 |
| slots-only prefix | 2/6 | 2/6 | 6/6 |

## Decision

Kill the low-capacity source cross-attention prefix-emitter family on this
surface. It has now failed under:

- plain gold-vs-distractor continuation training
- source-control contrastive training
- target-side continuation next-token CE training
- generation-time clean-ID evaluation

The failure mode is stable: the matched source prefix is weaker than
source-destroyed or target-only controls.

## Artifacts

| Artifact | SHA256 |
|---|---|
| `results/qwen25math_svamp32_target_ce_generation_gate_20260426/smoke.json` | `a8fda429e29ba9ed1ee06285706dc3f0fa95609be2f8829a508b379e689c9517` |
| `results/qwen25math_svamp32_target_ce_generation_gate_20260426/smoke.md` | `20be9a45f3be23453c75cd3b15d9a42953c5ae75cc59299e28ab2f19be3e22d9` |
| `results/qwen25math_svamp32_target_ce_generation_gate_20260426/generations.jsonl` | `f6cff21a9f981f6482f04b53ef9902cad6ea9ea079b05e5e57c88a03f998acad` |
| `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py` | `6c6b3fde3f4c2ecc12f071fb69e80c0f58b47559b1299ad4c2b740cbc4074a36` |

## Next Gate

Do not tune this prefix-emitter branch further. The next highest-value branch is
a non-prefix memory interface with a matched C2C-fuser baseline, or a new
source/surface discovery pass if no such interface can be implemented cheaply.

The exact next command should start from the reusable `latent_bridge` query-
innovation resampler path and either add true LM CE/generation scoring there or
prove that the translator API cannot support it without a larger refactor.
