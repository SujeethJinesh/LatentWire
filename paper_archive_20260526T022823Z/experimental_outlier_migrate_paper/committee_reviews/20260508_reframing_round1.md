# OutlierMigrate Committee Review: Related-Work Reframing Round 1

Reviewed draft:
`experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`

Reviewed reviewer pack:
`experimental/outlier_migrate/paper/reviewer_pack.md`

## A. COLM Area Chair

Scores: novelty 6/10, rigor 6/10, clarity 8/10.

Meta-review: The revised reframing mostly fixes the novelty boundary. The
draft explicitly says OutlierMigrate is not the first work on dynamic outliers
in Mamba-style models and scopes QMamba/OuroMamba to vision Mamba prior art.
This is supported by primary sources: QMamba reports highly dynamic
hidden-state sequences, and OuroMamba reports dynamic activation-outlier
variation across time steps with dynamic outlier detection during inference.

The paper is clear that the new contribution is narrower: hybrid LLM
long-reasoning traces under frozen Phase 0/1 gates. Static-protection methods
are treated fairly; the paper says they are not claimed to be wrong and only
motivate architecture-specific validation.

Fixable issues: soften "weakens the static-outlier hypothesis" to "on this
Granite rank-migration surface" everywhere it appears. Add one sentence
acknowledging that methods like KVQuant and BlockDialect already include
per-vector/online adaptation rather than being purely fixed static maps.

## B. MLSys Reviewer

The systems contribution remains unproven. The result is systems-relevant
because static channel protection and quantization policies may become stale
during long decode, but this paper measures rank migration only. It does not
yet provide a kernel, cache policy, quantization intervention, oracle headroom
study, or end-to-end throughput/quality tradeoff.

Reproducibility is the strongest part: the draft reports fixed model
snapshots, prompt hashes, activation artifact hashes, checker outputs,
artifact completeness, bootstrap seeds, and exact commands. The reviewer pack
also records the decision values and artifact paths.

Engineering rigor gaps are clear and fixable in the next experimental packet:
add one-command verification, hash-check scripts, capture-semantics tests,
rank-tie behavior, decode-position reachability checks, and layer-stratified
summaries. For MLSys, the required baseline is static top-channel protection
versus oracle or learned migration-aware protection, with paired uncertainty
and cost accounting.

One paper-text issue: the limitation paragraph mentions "the current
authorized work window" and vLLM compatibility. That is runbook/project
language, not submission prose. Replace it with neutral future-work language.

## C. Adversarial Reviewer

The reframing succeeds on the main requested point: I do not see a remaining
claim that this is the first dynamic-outlier result in Mamba. The draft
explicitly credits QMamba and OuroMamba, and the primary sources support that
credit. I also do not see obvious hallucinated citations from the spot-check:
Mamba-PTQ, Quamba, MambaQuant, SmoothQuant, AWQ, QuaRot, KVQuant,
BlockDialect, GLA, Gated DeltaNet, Kimi Linear, and Qwen3.6 sources exist and
broadly match the local descriptions.

The adversarial concern is statistical interpretation. A migration fraction
around 0.84 easily clears the preregistered 0.05 threshold, but without nulls
it may still reflect ordinary rank churn. Missing controls: random-channel
drift, shuffled-trace nulls, adjacent-position drift, same-position recapture
stability, layer-stratified bootstrap, sensitivity to top-1% and rank-delta
thresholds, and prompt-length/position reachability audits.

Static methods are mostly treated fairly, but "static-outlier hypothesis"
risks sounding broader than the evidence. The paper should not imply that
SmoothQuant/AWQ/QuaRot/KVQuant/BlockDialect would fail here without running
them or an oracle-protection proxy.

## Fix Response

Applied in the same round:

- Replaced broad "weakens the static-outlier hypothesis" phrasing with a
  Granite rank-migration surface statement.
- Added a fairness sentence noting that KVQuant and BlockDialect already
  contain adaptive components.
- Replaced runbook-style "authorized work window" wording in limitations with
  neutral future-validation language.
