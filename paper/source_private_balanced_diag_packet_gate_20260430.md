# Source-Private Balanced Diagnostic Packet Gate

- date: `2026-04-30`
- gate: `source_private_balanced_diag_packet_gate`
- artifacts: `results/source_private_diag_only_public_ablation_20260430/`
- status: pass at `n=500` over two seeds; promotes a cleaner source-causality
  contribution and demotes the previous semantic-view row as shortcut-prone

## Question

Can source-private bytes still carry useful information when public candidate
semantics and obvious diagnostic-code artifacts are removed as explanations?

Layman version: before, a smart public-only classifier could read the public
candidate wording and guess the right patch without the private clue. I changed
the task so all candidate diagnostic codes look plausible, then tested whether
a 2-byte private diagnostic clue still picks the right candidate while a
public-only classifier cannot.

## Method

I added:

- `diagnostic_table_mode="plausible_decoys"` in the hidden-repair benchmark.
  Distractor candidates no longer get obvious `X0/X1/...` codes; they get
  plausible two-character diagnostic codes that look like the true code.
- `candidate_view="diag_only"` for learned/public receiver surfaces. This view
  exposes only `handles_repair_diag=<code>` for each candidate.
- a separately trained public-only receiver:
  `scripts/run_source_private_public_only_receiver_ablation.py`.
- a paired summary:
  `scripts/summarize_source_private_balanced_diag_packet_gate.py`.

The decisive comparison uses the same `n=500` eval IDs:

- direct 2-byte source-private diagnostic packet;
- target-only and source-destroying controls;
- same-byte text/JSON/free-text negative controls;
- full-log/full-diagnostic positive oracles;
- a separately trained public-only diagnostic receiver with no packet bytes.

## Results

Aggregate artifact:
`results/source_private_diag_only_public_ablation_20260430/summary/`

Headline:

- pass gate: `True`
- runs: `2`
- budget bytes: `2`
- min packet accuracy: `1.000`
- max public-only accuracy: `0.178`
- min packet-public CI95 low: `+0.788`
- max public-target CI95 high: `-0.022`
- same eval IDs across packet/public rows: `True`

Rows:

| Seed | n | Packet | Public-only | Target | Best control | Packet-public CI |
|---|---:|---:|---:|---:|---:|---|
| 29 | 500 | 1.000 | 0.178 | 0.250 | 0.250 | `[0.788, 0.856]` |
| 31 | 500 | 1.000 | 0.142 | 0.250 | 0.250 | `[0.830, 0.888]` |

The direct diagnostic packet also passes all tested budgets (`2/4/8` bytes) on
both seeds. Same-byte text, truncated JSON, truncated free text, helper-only
templates, and diagnostic-masked logs stay at target. Full hidden logs and full
diagnostic text are positive oracles at `1.000`.

## Failed Probe

The learned masked-consistency receiver did not transfer to this harder
balanced `diag_only` surface:

- public-only n64: `0.281` vs target `0.250`;
- learned packet-conditioned n64: `0.312`;
- deterministic Hamming n64: `0.188`;
- pass gate: `False`.

Interpretation: the current learned syndrome encoder is not yet a robust
diagnostic-code transmitter under balanced decoys. That is a useful pruning
result, not a paper claim.

## Interpretation

This gives the paper a cleaner source-causality contribution than the previous
semantic-view receiver result:

> With public candidate semantics removed and all candidate diagnostic codes
> made plausible, a 2-byte source-private packet still identifies the correct
> candidate at `1.000` accuracy over two `n=500` seeds, while a separately
> trained public-only receiver fails below target.

This is not broad latent transfer. It is a rigorous side-information coding
result: the target has a public candidate table, and the source sends the
private residual key needed to resolve that table.

## Reviewer Impact

This directly addresses the strongest current reviewer attack:

- not public-semantics-only: public-only receiver fails;
- not obvious `X*` code artifacts: distractor codes are plausible;
- not same-ID leakage: public-only train/eval are disjoint and direct/public
  comparisons use the same eval IDs;
- not same-byte text relay: truncated text/JSON/free text fail at target;
- not target-cache: target-only/target-wrapper stay at `0.250`.

The right paper story is now narrower and stronger:

1. low-rate source-private diagnostic packet protocol;
2. learned masked-consistency receiver as a method-depth ablation on the
   original semantic side-information surface;
3. balanced diagnostic gate as the cleanest source-causality proof;
4. Mac systems accounting showing why byte-scale private packets are a useful
   operating point.

## Next Gate

Highest-value next gate:

```text
source_private_balanced_diag_packet_cross_family_or_model_20260430
```

Run the balanced diagnostic packet on one cross-family/model-mediated receiver
surface, or add a learned receiver that can recover the direct diagnostic-code
behavior without explicit string matching.

For ICLR comfort, the next missing pieces are:

- one cross-family/model pair;
- native serving telemetry when NVIDIA is available;
- a learned receiver that approaches the direct diagnostic packet on the
  balanced `diag_only` surface.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_masked_consistency_receiver_smoke.py \
  tests/test_run_source_private_public_only_receiver_ablation.py \
  tests/test_summarize_source_private_balanced_diag_packet_gate.py
```

Outcome: `10 passed in 3.31s`.
