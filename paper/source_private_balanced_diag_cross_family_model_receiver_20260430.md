# Balanced Diagnostic Cross-Family and Model-Receiver Gate

- date: `2026-04-30`
- artifacts:
  - `results/source_private_balanced_diag_cross_family_20260430/`
  - `results/source_private_balanced_diag_target_decoder_20260430/`
- status: cross-family direct/public gate passes; frozen Qwen target decoder is
  a useful partial but not a strict promotion

## Cycle Start

1. Current ICLR readiness and distance: stronger scoped full-paper candidate,
   still not comfortably ICLR-ready. The clean direct packet now survives
   cross-family public-only training/eval splits, but the model-mediated
   balanced receiver is still only partial.
2. Current story: the source sees private hidden evidence, sends a tiny
   diagnostic packet, and the target resolves a public candidate table. Gains
   must vanish under public-only and source-destroying controls.
3. Exact blocker: reviewers can still call the strongest positive a
   protocol/table result unless a learned or frozen target receiver passes the
   balanced surface at medium scale.
4. Current live branch: balanced plausible-decoy diagnostic packet with
   public-only and model-mediated receiver hardening.
5. Highest-priority gate: cross-family public-only falsification plus the
   cheapest frozen-target decoder probe on the same balanced surface.
6. Scale-up rung: large frozen diagnostic slice for direct/public; smoke for
   balanced model-mediated target decoding.

## Layman Version

The receiver has a public list of four possible fixes. The source privately
knows which hidden test failed and sends only a two-character clue. I tested
whether that clue still helps when the public list is made harder, when the
public-only receiver trains on different repair families, and when a small
Qwen model, rather than a hand-written rule, tries to use the clue.

## Harness Hardening

I made two reproducibility fixes before running the gate:

- `scripts/run_source_private_hidden_repair_packet_smoke.py` now exposes
  `--start-index` and records both `--start-index` and
  `--diagnostic-table-mode` in the replayable manifest command.
- `scripts/summarize_source_private_balanced_diag_packet_gate.py` now requires
  exact eval-ID parity plus family-name and answer-label parity, public
  train/eval ID disjointness, matching direct/public eval configuration, and
  `plausible_decoys` + `diag_only` configuration.

## Cross-Family Direct/Public Gate

Commands followed this structure for seeds `29` and `31`:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 --candidates 4 --seed 29 --budgets 2,4,8 \
  --family-set holdout --start-index 0 \
  --diagnostic-table-mode plausible_decoys \
  --output-dir results/source_private_balanced_diag_cross_family_20260430/direct_holdout_n500_seed29

./venv_arm64/bin/python scripts/run_source_private_public_only_receiver_ablation.py \
  --output-dir results/source_private_balanced_diag_cross_family_20260430/public_core_to_holdout_n500_seed29 \
  --train-examples 512 --eval-examples 500 \
  --train-seed 30 --eval-seed 29 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-family-set core --eval-family-set holdout \
  --diagnostic-table-mode plausible_decoys \
  --candidate-view diag_only --feature-dim 512 --require-no-leak
```

Summary artifact:
`results/source_private_balanced_diag_cross_family_20260430/summary/`

Headline:

- pass gate: `True`
- rows: `4`
- directions: `core->holdout`, `holdout->core`
- budget bytes: `2`
- min packet accuracy: `1.000`
- max public-only accuracy: `0.178`
- min packet-public CI95 low: `+0.788`
- max public-target CI95 high: `-0.022`
- exact eval IDs, family names, answer labels, direct/public config, and
  public train/eval disjointness: all pass

Interpretation: this is not model-family latent transfer. It is a stronger
side-information falsification: even when the public-only receiver trains on
different repair families, it cannot infer the private diagnostic; the 2-byte
source packet still resolves the table perfectly.

## Balanced Frozen Target Decoder

I ran `Qwen/Qwen3-0.6B` CPU decoding on the balanced `n=32` slice. MPS was
checked first but the model path failed with an MPS matmul shape error, so the
recorded runs are CPU-only.

Label-output prompt:

- artifact:
  `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_cpu/`
- point gate: `True`
- matched packet: `0.688`
- target-only: `0.250`
- best control: `0.250`
- matched-target delta: `+0.438`
- paired CI95 lower bound vs target and best control: `+0.281`
- valid prediction rate: `0.938`
- uncertainty summary pass: `False`, because the strict valid-rate threshold is
  `>=0.95`
- p50 matched latency: `2133.8 ms`

Choice-alias prompt:

- artifact:
  `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_choice_cpu/`
- point gate: `False`
- matched packet: `0.250`
- target-only/control rows: `0.250`
- valid prediction rate: `1.000`
- failure mode: the model collapses to option-letter priors (`A/C/D`) and does
  not condition on the packet.

Interpretation: the original label prompt is the better model-mediated
receiver, but the current balanced model row is not yet a promotion. The next
model-mediated step should improve the target receiver without making the
packet parser more permissive.

## Related-Work Implication

Recent activation and cache communication work makes broad latent-transfer
claims expensive to defend. Activation handoff and C2C/KVComm transfer much
higher-rate internal states, while this gate is an extreme-rate
decoder-side-information result. TurboQuant and KV compression sharpen the
systems comparison: our byte claim is strongest when the source cannot expose
full KV/cache state and only a boundary packet may cross.

## Decision

Promote:

- balanced direct diagnostic packet with public-only controls;
- cross-family public-only falsification for the balanced diagnostic surface;
- harness hardening as a reproducibility/control contribution.

Do not promote yet:

- choice-alias frozen target receiver;
- balanced model-mediated receiver at medium scale;
- broad latent-transfer or C2C-beating claims.

## Next Exact Gate

`source_private_balanced_diag_target_decoder_prompt_repair_20260501`

Run one stricter label-output prompt or constrained candidate-label receiver at
`n=64` first. Pass rule: matched packet beats target and best control by at
least `+0.15`, valid prediction rate `>=0.95`, paired CI95 lower bound `>+0.10`,
and no parser that maps raw packet-code outputs to candidate labels.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_target_decoder_smoke.py \
  tests/test_summarize_source_private_target_decoder_uncertainty.py \
  tests/test_run_source_private_hidden_repair_packet_smoke.py \
  tests/test_summarize_source_private_balanced_diag_packet_gate.py
```

Outcome: `20 passed in 0.26s`.
