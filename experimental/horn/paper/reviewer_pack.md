# HORN Reviewer Pack

- status: preregistered Mac-gate scaffold; no positive asymmetry result
- current decision: blocked on real hybrid boundary activation dumps
- camera-readiness: not submittable as a paper result until H1--H3 have real
  trace evidence

## Paper Link

- Draft PDF: `experimental/horn/paper/horn_colm2026.pdf`
- Draft TeX: `experimental/horn/paper/horn_colm2026.tex`
- Preregistration: `experimental/horn/phase2/preregister_horn_20260506.md`

## Current Claim

HORN tests whether `attention->ssm` and `ssm->attention` boundaries differ in
activation-outlier magnitude and FP4-equivalent noise sensitivity. The current
artifacts define the gate and controls; they do not claim directional
asymmetry.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | H1/H2 should use reasoning traces plus WikiText/GSM-style drift controls, but no live hybrid activations have been dumped yet. | Gate pending. |
| Ablations | Required controls are explicit boundary maps, non-boundary adjacent pairs, matched normalization placement, direction-label permutation, and pure-architecture controls. | Adequate before real H1. |
| Correctness | The checker now requires at least 12 prompts unless resource-limited, both boundary directions, both-direction non-boundary controls, finite numeric fields, 64-hex prompt/architecture provenance, recomputed H1 `summary.json` gate aggregates, and each permuted-direction row matching an observed prompt/boundary/norm tuple while flipping direction. | Artifact path is hardened. |
| Reproducibility | Synthetic H1 packet is deterministic, and shared architecture maps fix boundary IDs. | Not model evidence. |
| Novelty | The proposed wedge is directional propagation through hybrid boundaries, not generic activation-outlier measurement. | Plausible only if H1/H2/H3 pass. |
| Camera-readiness | The draft is a preregistration shell. It needs real H1/H2/H3 tables before submission as a method or measurement paper. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic H1 | SSM-to-attention / attention-to-SSM max ratio 3.775, kurtosis ratio 7.139 | validates readout only |
| architecture provenance | shared boundary IDs and direction counts exist | packet provenance ready |
| model eligibility | live targets are identified, but weights are not cached locally | blocked on model load |
| real-packet checker | rejects missing directions, stale summary fields, too few prompts, non-finite rows, promotable resource-limited decisions, unpaired permuted controls, and permuted controls that preserve the selected directional effect | ready for real H1 |

The required matched flipped controls are the `permuted_direction` rows: they
must reuse an observed boundary tuple with the same prompt ID and normalization
positions, then invert only the direction label. The H1 evaluator records
selected-direction control ratios, so it rejects controls that keep the
high-magnitude signal on the same direction label. A faithful label flip that
preserves unsigned max/min asymmetry but moves the signal to the opposite label
is treated as an acceptable null.

## Reviewer Risks

- No real boundary activations have been measured.
- Directional ratios may vanish under pure-architecture controls.
- HORN may collapse into HBSM if the effect is just layer sensitivity.
- A precision-allocation recipe must eventually beat uniform precision at
  matched memory.

## Next Exact Gate

Run H1 on the smallest available live hybrid model. The first packet must pass:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_gate_h1_<YYYYMMDD>_<model_slug> \
  --mode real --project horn
```

Continue only if real boundary-direction asymmetry passes the preregistered H1
rule.
If the run is resource-limited, record it with
`RESOURCE_LIMITED_NOT_PROMOTABLE` and do not treat it as H1 promotion.
