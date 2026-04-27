# Creative Reference Synthesis

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; no live positive method
   has survived strict source controls, seed stability, and cross-family
   falsification.
2. Current paper story: historical RotAlign/latent-bridge results are mechanism
   clues, but the recent source-likelihood and syndrome branches are killed by
   controls. The next story must be source-derived side information decoded by
   the target, not target priors with a source-shaped wrapper.
3. Exact blocker to submission: no branch currently recovers source-necessary
   wins beyond target-alone/text/C2C controls; MPS is also blocked by orphaned
   PID `31103`.
4. Current live branch or top candidates: no live branch; top candidates are
   candidate-syndrome decoding, zero-init gated query bottlenecks, and
   anchor-relative sparse difference atoms.
5. Highest-priority gate: run the CPU-first candidate-syndrome decoder over
   existing SVAMP70 artifacts, then use MPS only after the blocked process is
   cleared.
6. Scale-up rung: source-surface/new-branch discovery, pre-smoke.

## What Changed

- Converted 172 local reference PDFs into markdown under
  `references/pdf_markdown/`.
- Added a durable conversion manifest with source PDF hashes, page counts, and
  extraction status.
- Promoted four method families from the reference sweep:
  candidate-syndrome decoding, zero-init gated query bottlenecks,
  anchor-relative sparse difference atoms, and protected-tail quantized
  residual systems ablations.

## Evidence Update

The literature sweep weakens another shallow source-likelihood sketch. The best
fit to the current blocker is a side-information view: the source should send a
tiny code that helps the target choose among its own candidates. That creates
natural controls: random syndrome, shuffled source, zero source, target-only,
slots-only, and matched byte budget.

Connector literature argues for target-preserving injection if a learned branch
is needed: fixed query tokens or a resampler should be zero-gated into the
target so target-alone behavior is the starting point, not something repaired
after damage.

Representation-geometry literature argues against another global RotAlign
tweak. A revived geometry branch should be anchor-relative/local and should
separate shared atoms from source-difference atoms so controls can remove only
the claimed communication lane.

## Branch Decisions

- Promoted: candidate-syndrome decoder as the next CPU-feasible branch.
- Promoted: zero-init gated query bottleneck as the next learned branch once
  MPS clears.
- Revived with constraints: RotAlign/latent-bridge only through
  anchor-relative sparse difference atoms.
- Deferred: protected-tail quantized residuals until a branch has real
  source-necessary wins worth compressing.
- Weakened: shallow source-likelihood and generic Perceiver memory variants.

## CPU Gate Result

Command:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir results/candidate_syndrome_decoder_20260427 \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --run-date 2026-04-27
```

Result: `candidate_syndrome_decoder_fails_smoke`.

- Live: matched clean source-necessary `1`, matched target-self harms `17`,
  control clean union `0`.
- Holdout: matched clean source-necessary `4`, matched target-self harms `14`,
  control clean union `0`.

Decision: do not promote the numeric hash-syndrome artifact probe. The
candidate-syndrome family remains viable only with learned source predicates or
a stronger source surface.

## Next Exact Gate

Before any MPS command, recheck:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`. If the scout fails source
headroom, move directly to the zero-init gated query bottleneck once a viable
source surface is available.
