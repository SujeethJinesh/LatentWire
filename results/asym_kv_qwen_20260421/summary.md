# Separable Asymmetric K/V Qwen Smoke

Date: 2026-04-21

Checkpoint:

`checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`

Model pair:

- Source: `Qwen/Qwen2.5-0.5B-Instruct`
- Target: `Qwen/Qwen3-0.6B`

Common settings:

- `--source-use-chat-template --target-use-chat-template`
- `--source-enable-thinking false --target-enable-thinking false`
- `--source-reasoning-mode brief_analysis`
- `--gate-mode fixed --fixed-gate 0.10`
- `--fusion-rule static --kv-transport both`
- `--kv-route-selection-ratio 0.25 --kv-value-selection-ratio 0.75`

## Results

| Split | Route metric | Value metric | Target | RotAlign | Delta | Avg bytes | Route/value overlap | Jaccard |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GSM5 | attention | energy | 0.20 | 0.40 | +0.20 | 1369158.0 | 0.674 | 0.206 |
| GSM5 | random | random | 0.20 | 0.20 | 0.00 | 1369158.0 | 0.837 | 0.270 |
| GSM10 | attention | energy | 0.10 | 0.10 | 0.00 | 1366231.3 | 0.666 | 0.202 |

## Interpretation

The new metric-separated selector fixes the prior nested-mask failure: route
and value masks are no longer identical, and the attention/energy run shows
lower overlap than the random/random control. The GSM5 result has one
method-only win, but GSM10 is neutral, so this is a promising ablation lane
rather than a settled positive method.

Next run should scale to GSM30 with three matched controls:

- route attention / value energy
- route attention / value attention
- route random / value random

## GSM30 Paired-Telemetry Result

Command output:

| Split | Route metric | Value metric | Target | RotAlign | Delta | Avg bytes | Method-only | Baseline-only | Both correct | Both wrong |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM30 | attention | energy | 0.0667 | 0.0667 | 0.0000 | 1,406,292.3 | 1 | 1 | 1 | 27 |

Sidecar:

`qwen_gsm30_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat_telemetry.jsonl.meta.json`

Non-neutral paired examples:

| Index | Example id | Flip | Method answer | Baseline answer | Bytes | Route/value overlap | Jaccard | Source/target token ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | `c11b1f65a91b1796` | method-only | 20 | 25 | 2,127,972.0 | 0.697 | 0.213 | 1.189 |
| 13 | `c1d4c219268d7f10` | baseline-only | 2 | 20 | 1,635,392.5 | 0.659 | 0.196 | 1.246 |
| 17 | `d750c66e733a2837` | both-correct | 3 | 3 | 1,172,272.5 | 0.664 | 0.198 | 1.341 |

Interpretation:

The GSM30 result does not replicate the GSM5 positive signal. The value of
this run is the telemetry: the evaluator can now identify example-level flips
and associate them with byte cost, token-count ratio, and route/value selector
structure. The current method should stay in the ablation lane until matched
GSM30 controls show a reliable method-only excess over baseline-only.

## GSM30 Matched Selector Controls

All rows use the same Qwen pair, checkpoint, gate `0.10`, route budget `0.25`,
value budget `0.75`, and average bytes `1,406,292.3`.

| Route metric | Value metric | Target | RotAlign | Delta | Method-only | Baseline-only | Both correct | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attention | energy | 0.0667 | 0.0667 | 0.0000 | 1 | 1 | 1 | 27 |
| attention | attention | 0.0667 | 0.1000 | +0.0333 | 2 | 1 | 1 | 26 |
| random | random | 0.0667 | 0.1333 | +0.0667 | 4 | 2 | 0 | 24 |

Interpretation:

The best GSM30 smoke in this family is currently `random/random`, not the
semantic route-attention/value-energy selector. That means the current positive
signal cannot be claimed as a selector-quality result. It is more consistent
with one of these hypotheses:

- cache/message perturbation sometimes helps generation escape target-alone
  failures;
- attention/energy selection is overconstrained at this small model scale;
- the gate/fusion path matters more than the current selector metric;
- stochastic or ensemble-style routing may be a better lead than deterministic
  attention-only routing.

Next selector-specific claim needs random-seed repeats and controls that hold
the perturbation distribution constant while changing only selector semantics.

## Protected-Channel Toy Result

Artifact:

`../query_pool_toy_20260421/query_pool_protected_channel_residual_codebook_vs_topk.md`

| Scenario | Codebook | Residual codebook | Protected residual codebook | Protected delta vs residual |
|---|---:|---:|---:|---:|
| aligned | 0.4948 | 0.5417 | 0.5938 | +0.0521 |
| rotated | 0.5781 | 0.6406 | 0.5729 | -0.0677 |
| outlier | 0.5312 | 0.5677 | 0.6198 | +0.0521 |
| slot-permuted | 0.4688 | 0.5417 | 0.4948 | -0.0469 |

Interpretation:

Protected channels are useful when the protected subspace is aligned or carries
true outlier energy, but brittle under rotation and slot permutation. This is a
clean blocker: the paper method cannot rely on fixed protected coordinates
unless the bridge first solves gauge/permutation alignment or learns the
protected mask.
