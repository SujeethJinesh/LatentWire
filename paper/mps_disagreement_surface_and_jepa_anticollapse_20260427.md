# Disagreement Surface And JEPA Anti-Collapse Cycle

- date: `2026-04-27`
- readiness: not ICLR-ready
- rung: micro smoke / selected discovery
- live branch: answer-masked source-surface discovery; JEPA-inspired connector
  design only after non-leaky headroom exists

## Cycle Start

1. Current ICLR readiness: not ready; no method survives strict controls.
2. Current paper story: MPS is usable again, stronger-source 7B surfaces create
   some source/target disagreement, but answer-masking still kills promotion.
3. Exact blocker: find source-only target-pool headroom that is not explained by
   source final/verified numeric answers.
4. Current candidate branches: 7B disagreement-slice discovery; JEPA-inspired
   answer-masked latent prediction as the next connector objective if a clean
   surface appears.
5. Highest-priority gate: audit selected 6-ID and 12-ID disagreement slices for
   answer-unexplained clean-in-pool IDs.
6. Scale-up rung: selected micro discovery, not paper-eligible.

## Disagreement Slices

### Historical Clean6 Probe

- artifact: `results/mps_qwen25_7b_historical_clean6_discovery_20260427/`
- eval source: `data/svamp_eval_70.jsonl`
- selected IDs: historical `clean_source_only` from
  `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json`
- target: `0/6`, numeric coverage `6/6`
- source: `1/6`, numeric coverage `6/6`
- text relay: `1/6`, numeric coverage `6/6`
- clean source-only after text relay: `1`
- answer-unexplained clean in pool: `0`

Decision: fail. The only source rescue is not in the target-side candidate pool
and is explained by source final answer `2`.

### Disagreement12 Union Probe

- artifact: `results/mps_qwen25_7b_disagreement12_discovery_20260427/`
- eval source: `data/svamp_1000.jsonl`
- selected IDs: union of historical live clean6, chal241-310 clean4, and holdout
  clean2 recommended by the artifact-audit subagent
- target: `0/12`, numeric coverage `12/12`
- source: `4/12`, numeric coverage `12/12`
- text relay: `4/12`, numeric coverage `12/12`
- source-only over target: `4`
- clean source-only after text relay: `2`
- clean in target-side pool: `1`
- answer-unexplained clean in pool: `0`

Decision: fail for positive-method promotion. The two clean source-only IDs are:

- `ab1e71e8928661d0`: gold answer is in the target-side pool, but the source
  final answer is exactly the gold value `4`.
- `ce08a3a269bf0151`: source final answer is exactly the gold value `2`, and
  the gold value is not in the target-side candidate pool.

The surface is useful as a harder disagreement diagnostic, not as evidence of
nontrivial communication.

## JEPA / Anti-Collapse Synthesis

The JEPA literature changes the next connector design but not the promotion
status. I-JEPA and V-JEPA motivate predicting frozen target latents under hard
masking rather than reconstructing surface tokens. LeJEPA/SIGReg, VICReg, and
Barlow Twins motivate explicit collapse telemetry: sideinfo variance, covariance
off-diagonal mass, effective rank, and cross-view redundancy. LLM-JEPA and
VL-JEPA motivate language-side latent prediction and answer-embedding/verifier
diagnostics before full decoding.

Concrete translation:

- create answer-masked dual source views before extracting sideinfo
- predict stopped/frozen target latent/KV targets
- require matched-source answer-likelihood margin over zero-source,
  shuffled-source, target-only, slots-only, and answer-only controls
- add target-preservation loss on target-correct IDs
- report bytes/rate plus effective-rank and covariance collapse telemetry

Do not run this connector on the current surfaces as a headline method: stored
and fresh surfaces still have `answer_unexplained_clean_in_pool = 0`.

## Decision

- passed: MPS remains clear; selected 6-ID and 12-ID slices ran with exact ID
  parity and high numeric coverage.
- failed: 7B disagreement slices still do not reveal answer-unexplained
  target-pool headroom.
- revived only as design: JEPA anti-collapse objectives for a future connector,
  contingent on first finding a non-leaky source surface.

## Next Exact Gate

Implement CPU-only answer-masked source/answer-only diagnostics before another
MPS generation sweep:

1. Add or reuse `answer_masked_source` and `answer_only` controls for source
   sideinfo extraction.
2. Add collapse telemetry for sideinfo vectors: variance floor, effective rank,
   covariance off-diagonal, Barlow diagonal/off-diagonal.
3. Run an answer-likelihood smoke on the live and holdout source surfaces,
   requiring matched source to beat zero/shuffled/target-only/slots-only and
   answer-only controls with no target-preservation harm.
