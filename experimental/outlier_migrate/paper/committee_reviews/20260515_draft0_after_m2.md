# Committee Review - Draft 0 After Phase 9 M2

## COLM Area Chair

Score: 7/10.

The draft is now coherent as a measurement and mechanism paper. The title and
abstract correctly pivot away from a Mamba-specific claim toward
decode-position channel drift, and the evidence base is broader than before:
Granite Phase 0/1, partial Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1 are
all integrated. The strongest contribution is the strict set-leaving
decomposition, because it converts a rank-drift observation into a
static-protection failure mode. Rigor is helped by the fact that failed method
attempts are reported honestly, especially M2's
`KILL_M2_RANDOM_CONTROL_BEATS` result in
`experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z/checker_result.json`.
The main weakness is that the paper is not yet a positive-method paper. For
COLM workshop, the characterization and negative method evidence are enough to
start reviewer discussion. For a main conference, the paper needs either M10/M9
to produce a positive method or a much tighter argument that failed static
methods are themselves the contribution.

## MLSys Reviewer

Score: 6/10.

The systems motivation is real but the systems contribution is still mostly
diagnostic. The paper now does the right thing by not claiming latency,
throughput, memory, or quality improvement. The Phase 3/4/M2 failures are
useful because they rule out simple protected-set recipes under W4A16, and M2
has an important matched negative control: random-bin assignment beats the
intended position-conditioned assignment by `0.667548290168` median recovery.
However, the draft still lacks a working systems intervention, a cost model, or
a runtime policy. The reproducibility section is much improved: artifact paths
for Phase 4, Phase 5', Phase 7, Step 9.0, and M2 are listed. The strongest
engineering next step is M10 or M9: either show that scale refresh/prediction
captures the drift, or prove that the drift is hard to exploit. The draft is
credible for a workshop, but an MLSys/ICLR systems claim remains blocked.

## Adversarial Reviewer

Score: 7/10.

The new draft avoids the most dangerous overclaim: it does not call M2 a
method success. It also no longer over-centers Mamba, which would have been
untenable after DeepSeek and Falcon-H1. The main residual risks are scope and
selection. The vacation-mode 12-trace M2 slice is documented, but the paper
should keep reminding readers that M2 is a reduced-slice negative result after
OOM adaptation. The strict set-leaving decomposition is post-hoc; it is
convincing but must remain separate from preregistered gate decisions. The
related-work claims about Kimi Linear and Qwen3.6 must remain theoretical
grounding only, since those models were not measured. I do not see p-hacking in
the current draft: thresholds and failed interventions are reported plainly,
and the random-control kill is not buried. The biggest missing item is still a
complete citation/contamination audit before submission.
