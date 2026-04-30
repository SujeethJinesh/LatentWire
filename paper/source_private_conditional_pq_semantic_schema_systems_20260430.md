# Conditional PQ Semantic/Schema/Systems Stress

- date: `2026-04-30`
- status: strengthens the live bounded positive method; does not close
  cross-family ICLR gate
- references: `references/544_conditional_pq_semantic_schema_systems_refs_20260430.md`
- artifacts:
  - `results/source_private_conditional_pq_innovation_gate_20260430/`
  - `results/source_private_conditional_pq_basis_schema_grid_20260430/`
  - `results/source_private_conditional_pq_packet_isa_waterfall_20260430/`
- code:
  - `scripts/run_source_private_conditional_pq_innovation_gate.py`
  - `scripts/summarize_source_private_conditional_pq_innovation_gate.py`
  - `scripts/summarize_source_private_conditional_pq_basis_schema_grid.py`
  - `scripts/build_source_private_conditional_pq_packet_isa_waterfall.py`

## Purpose

The prior conditional PQ innovation row showed disjoint-ID success, but the
strongest reviewer objection was that `shared_text` might still rely on direct
public diagnostic handles. This stress asks two sharper questions:

1. Does the method still pass when direct diagnostic handles are removed from
   the candidate basis?
2. Does any existing public basis or diagnostic-table mode rescue
   bidirectional held-out-family transfer?

Layman version: first remove the obvious answer code from the public table and
see whether the tiny source packet still points to the right repair idea. Then
try many public coordinate systems on new bug families and see if any of them
make the packet understandable.

## Commands

Representative semantic no-diagnostic row:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_source_private_conditional_pq_innovation_gate.py \
  --output-dir results/source_private_conditional_pq_innovation_gate_20260430/n500_no_diag_remap101_uph_constrained_label \
  --train-examples 768 --eval-examples 500 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set all --eval-family-set all \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view no_diag \
  --source-topk 64 --target-topk 32 \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 --bootstrap-samples 1000
```

Cross-family basis/schema grid:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python - <<'PY'
from pathlib import Path
from scripts.run_source_private_conditional_pq_innovation_gate import run_gate

root = Path("results/source_private_conditional_pq_basis_schema_grid_20260430")
basis_views = ["shared_text", "anchor_relative", "semantic", "diag_only", "no_diag", "full", "slot"]
modes = ["plausible_decoys", "legacy"]
directions = [("core_to_holdout", "core", "holdout"), ("holdout_to_core", "holdout", "core")]
for mode in modes:
    for basis in basis_views:
        for name, train_family, eval_family in directions:
            run_gate(
                output_dir=root / f"{name}_{basis}_{mode}_n256",
                train_examples=768,
                eval_examples=256,
                train_start_index=10000,
                eval_start_index=0,
                train_seed=30,
                eval_seed=29,
                train_family_set=train_family,
                eval_family_set=eval_family,
                diagnostic_table_mode=mode,
                candidates=4,
                feature_dim=512,
                anchor_count=128,
                basis_view=basis,
                source_topk=64,
                target_topk=32,
                budget_bytes=4,
                variant="utility_protected_hadamard",
                remap_slot_seed=101,
                ridge=1e-2,
                fit_intercept=False,
                mask_repeats=1,
                codebook_iterations=12,
                seed=30,
                bootstrap_samples=1000,
            )
PY
```

Summaries:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_conditional_pq_basis_schema_grid.py \
  --grid-records results/source_private_conditional_pq_basis_schema_grid_20260430/summary_grid_records.json \
  --output-dir results/source_private_conditional_pq_basis_schema_grid_20260430/summary

./venv_arm64/bin/python scripts/build_source_private_conditional_pq_packet_isa_waterfall.py \
  --conditional-summary results/source_private_conditional_pq_innovation_gate_20260430/summary/conditional_pq_innovation_summary.json \
  --waterfall results/source_private_pq_transport_receiver_waterfall_20260430/pq_transport_receiver_waterfall.json \
  --output-dir results/source_private_conditional_pq_packet_isa_waterfall_20260430
```

## Results

The updated conditional PQ summary has `20` rows:

- decisive same-family n500 rows: `16/16` pass
- less-diagnostic n500 rows: `8/8` pass
- budget-2 n500 rows: `4/4` pass
- cross-family rows in the original conditional summary: `0/2` pass
- minimum decisive source accuracy: `0.996`
- maximum decisive best-control accuracy: `0.302`
- minimum decisive CI95 lower bound versus best control: `+0.658`

Less-diagnostic rows:

| Surface | Basis | Bytes | Remaps | Source | Target | Best control | CI95 low |
|---|---|---:|---|---:|---:|---:|---:|
| n500 all->all | semantic | 4 | 101/103/107 | 1.000 | 0.250 | 0.268-0.278 | >= +0.680 |
| n500 all->all | no_diag | 4 | 101/103/107 | 1.000 | 0.250 | 0.254 | +0.706 |
| n500 all->all | semantic | 2 | 101 | 1.000 | 0.250 | 0.302 | +0.658 |
| n500 all->all | no_diag | 2 | 101 | 1.000 | 0.250 | 0.290 | +0.668 |

Cross-family basis/schema grid:

- rows: `28`
- pass rows: `0`
- bidirectional basis/mode passes: `0`
- maximum source accuracy: `0.316406`
- maximum source minus best control: `+0.007812`
- maximum CI95 lower bound versus best control: `0.000`

This means existing static public bases do not rescue unseen-family transfer,
even though target-innovation oracle headroom stays high. The failure is not
just a bad choice among `shared_text`, `semantic`, `no_diag`, `diag_only`,
`full`, `slot`, or `anchor_relative`.

## Systems Waterfall

The conditional packet-ISA waterfall passes and attaches the live method rows
to the existing Mac-local transport/receiver measurements:

| Row | Acc min | Target | Best ctrl | CI95 low | Record B | Line B/req | DMA B/req | p95 ns | recv p50 ms | Text? | KV? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B conditional innovation packet | 1.000 | 0.250 | 0.302 | +0.658 | 5 | 5.0 | 6.0 | 0.688 | 0.01628 | false | false |
| 4B conditional innovation packet | 0.996 | 0.250 | 0.278 | +0.680 | 7 | 7.0 | 8.0 | 0.661 | 0.01628 | false | false |
| query-aware private text | | | | | 14 | 14.0 | 14.0 | 0.692 | | true | false |
| full hidden-log relay | | | | | 370 | 370.0 | 370.0 | 5.652 | | true | false |
| QJL 1-bit KV floor | | | | | 21504 | 21504.0 | 21504.0 | 404.175 | | false | true |
| KIVI/KVQuant 2-bit KV floor | | | | | 43008 | 43008.0 | 43008.0 | 1169.556 | | false | true |

This supports a systems claim about boundary traffic, private-state exposure,
and exact resident receiver decode on Mac. It is not a measured GPU serving,
HBM, PCIe/NVLink, or energy result.

## Interpretation

Promote:

1. Conditional innovation works without direct diagnostic handles on
   same-family/shared-schema rows.
2. The 2-byte packet is strong enough for semantic/no-diagnostic n500 rows.
3. Existing static bases fail cross-family; this gives a clear next-method
   target rather than another vague negative result.
4. The systems story now directly covers the live method: 5-byte and 7-byte
   packet records, no source text/KV exposure, Mac-local transport and receiver
   accounting.

Do not overclaim:

- This still is not unseen-family latent transfer.
- This still is not model-mediated LLM consumption of the packet.
- This still is not native GPU/vLLM serving speedup.
- Product quantization, rotations, and side-information coding are prior
  ingredients; the contribution is their use as a source-private conditional
  communication protocol with strict controls.

## Next Gate

The next method branch should be public-conditioned rather than static-basis:

- QINCo-style public-conditioned residual codebooks, where candidate
  innovations and target prior derive the codebook locally; or
- receiver-conditioned slots trained to preserve conditional innovation rather
  than source reconstruction.

Pass bar: bidirectional held-out family at n256 first, then n500/remap
confirmation; controls must stay near target and CI95 low versus best control
must exceed `+0.10`.
