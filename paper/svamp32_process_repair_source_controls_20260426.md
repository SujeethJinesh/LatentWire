# SVAMP32 Process Repair Source-Control Gate

- date: `2026-04-26`
- status: `process_repair_source_stack_fails_gate`
- scale-up rung: strict small exact-ID gate
- live branch entering gate: process-repair / selector stack on SVAMP32 clean residual surface
- decision: kill this branch as a source-communication method

## Start Status

- ICLR readiness: not ready
- current story: C2C and target-side repair expose headroom, but deployed
  LatentWire rows are still failing source-necessary controls or stability gates
- blocker: no source-derived signal recovers clean C2C-only IDs while preserving
  target-self repair wins
- highest-priority gate: test whether process repair on selected query-pool
  transport routes can recover clean residual IDs beyond target-self repair

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/process_repair_routes.py \
  --inputs .debug/svamp32_process_repair_source_controls_20260426/inputs/matched_target_plus_candidate.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --method rotalign_kv_gate_0.10 \
  --baseline-method target_alone \
  --selection-policy target_on_strict_format \
  --control-arms selected_no_repair \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --max-new-tokens 96 \
  --use-chat-template \
  --no-enable-thinking \
  --output-jsonl .debug/svamp32_process_repair_source_controls_20260426/outputs/matched_process_repair.jsonl \
  --output-md .debug/svamp32_process_repair_source_controls_20260426/outputs/matched_process_repair.md
```

## Result

| row | correct | notes |
|---|---:|---|
| target alone | `8/32` | fixed exact-ID target baseline |
| selected route, no repair | `8/32` | selected target on every example |
| process repair selected route | `10/32` | below target-self repair |
| target-self repair baseline | `14/32` | existing exact-ID artifact |

Clean residual accounting against
`results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`:

- clean residual recovered by process repair: `1/6`
- clean source-necessary recovered beyond target-self: not promotable
- target-self preservation: `1/3`
- target-self losses: `2/3`
- selected candidate source: `target` on `32/32`

Because the selector chose the target candidate for every example, the matched
row is target-side repair, not source communication. Running zero-source and
shuffled-source generations would not rescue the claim: the matched row already
fails the live gate, fails the target-self comparator, and has no selected
source signal.

## Artifact Hashes

- matched process repair JSONL:
  - `results/svamp32_process_repair_source_controls_20260426/matched_process_repair.jsonl`
  - sha256: `a125ac97e1e4c54763739cdcc23e13c8633a8dceb431c3bfbed6c351e5219f6d`
- matched process repair markdown:
  - `results/svamp32_process_repair_source_controls_20260426/matched_process_repair.md`
  - sha256: `98cf77da5299733d35f5618dd43815b6f90d3dc8904c561835351355b8797dfc`
- raw log:
  - `.debug/svamp32_process_repair_source_controls_20260426/logs/matched_process_repair.log`
  - sha256: `51e0f9286a25fce982dc6fd3de3ae2145bfd495fdc7c6f2018252da597ef1eb4`
- paired input JSONL:
  - `.debug/svamp32_process_repair_source_controls_20260426/inputs/matched_target_plus_candidate.jsonl`
  - sha256: `977ddd42bb999ac0e59bf25d687a1c5be997280e976d1b402d6effca1d91704c`

## Literature Update

Added `references/458_repair_verifier_source_control_refs.md`. The useful
takeaway is not to promote repair as source communication by itself. Any future
repair stack must be budget-matched against target-only self-consistency,
target self-refine, verifier/tool-only repair, and source-identity/order-shuffle
controls.

## Hypothesis Update

- killed: process-repair / selector stack as a source-communication method on
  the current SVAMP32 query-pool route surface
- weakened: further process-repair tuning without a source-aware selector
- promoted next: source-surface discovery through the cross-family
  tokenizer/interface stress branch, especially quotient/GPA sparse dictionary
  plus sequence-aligned byte sidecar

## Next Exact Gate

Run the smallest real cross-family tokenizer/interface stress gate that reuses
the quotient/GPA sparse dictionary and sequence-aligned byte sidecar evidence.
The first rung should check exact-ID parity, numeric coverage, source-destroying
controls, bytes/latency, and whether any source-derived sidecar improves over
target-only and text/token relay before scaling.
