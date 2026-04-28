# Source-Private Codebook-Remap Gate

- date: `2026-04-28`
- status: passed deterministic reviewer-risk ablation
- scale rung: large frozen deterministic ablation
- result root: `results/source_private_codebook_remap_gate_20260428/`

## Current Readiness

The scoped source-private diagnostic-packet artifact remains upload-ready. This
gate strengthens the ICLR case by reducing the fixed-codebook objection, but it
does not by itself upgrade the paper to a broad cross-architecture or MoE claim.

## Blocker Addressed

A skeptical reviewer can argue that the benchmark is a fixed label lookup: the
source emits a known two-character code, and the target maps that code to a
candidate. This gate asks whether the result survives codebook remapping while
holding the public task surface fixed.

## Method

I ran the deterministic hidden-repair packet harness with:

```bash
./venv_arm64/bin/python scripts/run_source_private_codebook_remap_gate.py \
  --examples 500 \
  --candidates 4 \
  --family-set all \
  --seeds 29,31,37 \
  --budgets 2,4,8,16 \
  --output-dir results/source_private_codebook_remap_gate_20260428
```

The harness keeps exact example IDs, family sequence, answer labels, and public
candidate labels fixed across seeds, but changes the diagnostic code assignment
and candidate diagnostic handles. It then reuses the strict packet/relay/control
conditions from the hidden-repair gate.

## Result

- pass gate: `true`
- examples per seed: `500`
- seeds: `29`, `31`, `37`
- budgets: `2`, `4`, `8`, `16`
- exact ID parity across seeds: `true`
- public surface parity across seeds: `true`
- unique codebooks: `3`

Across every seed and budget:

- matched repair packet accuracy: `1.000`
- best no-source accuracy: `0.250`
- best source-destroying control accuracy: `0.250-0.256`
- best reviewer-negative control accuracy: `0.250`
- minimum positive oracle accuracy: `1.000`
- structured JSON/free-text relays at the same low-rate budget: `0.250`
- diagnostic-masked full log: `0.250`

Representative diagnostic previews:

- seed `29`: `G0, H1, J2, K3, L4, M5, ...`
- seed `31`: `J0, K1, L2, M3, N4, P5, ...`
- seed `37`: `Q0, R1, S2, T3, U4, V5, ...`

## Interpretation

This supports the claim that the communication mechanism is not tied to one
fixed diagnostic vocabulary. It still remains a protocol-level diagnostic code:
the target has side information that maps codes to candidate handles. The right
paper wording is therefore:

> The method transfers source-private diagnostic evidence through a compact,
> remappable packet code under decoder side information.

Do not overstate this as natural-language semantic generalization or learned
latent transfer.

## What It Rules Out

- Fixed-codebook memorization as the sole explanation.
- Same-byte JSON or free-text relay explaining the low-rate result.
- Full-log formatting artifacts when the diagnostic field is masked.
- Target-prior, target-wrapper, random, shuffled, answer-only, or
  target-derived packet controls explaining the gain.

## What It Does Not Rule Out

- The task still exposes candidate-side diagnostic handles.
- The target decoder is mostly protocol-shaped.
- Architecture breadth is still incomplete.
- MoE and FP8 evidence are still unrun.
- Real tool traces may have noisier diagnostic extraction than this controlled
  benchmark.

## ICLR Impact

This should appear as a reviewer-risk ablation in the paper or appendix. It is a
stronger support row for the existing positive method, not a new headline claim.
The full-paper upgrade still needs:

1. Qwen3.6-35B-A3B n32 under identical controls.
2. Qwen3.6-35B-A3B-FP8 n32 under identical controls.
3. One non-Qwen current small model at n64/n160 without copied-helper if
   feasible.
4. A prompt-contract grid that reports valid-packet failure modes.
5. A systems/rate table that explicitly shows the low-rate packet region where
   structured text has not yet revealed the diagnostic code.

## Artifacts

- `results/source_private_codebook_remap_gate_20260428/summary.json`
- `results/source_private_codebook_remap_gate_20260428/summary.md`
- `results/source_private_codebook_remap_gate_20260428/manifest.json`
- `results/source_private_codebook_remap_gate_20260428/manifest.md`
