# Vacation Decision: M33 and M16 Combination Methods Authorization

Date: 2026-05-19 UTC

## Situation

Two recent results changed the Phase 9 method queue:

- M11b budget scaling mechanically passed on Granite-4-H-Small at the top-5
  EMA budget, with median recovery `0.4492840911245966`, but the CI95
  `[-1.3009000187907436, 1.00079062654749]` is too wide to treat as a
  settled positive method without replication.
- M26 stable-core protection returned `AMBIGUOUS_M26`, but showed real signal:
  stable-core median recovery `0.17761580764776214` versus random matched-core
  median `-1.549468781806685`.

The combination pattern suggests that stable structural channels and
EMA-smoothed dynamic protection may be complementary. The human authorized
two conditional follow-up methods, M33 and M16, while preserving KL
accumulation, ParoQuant, and M11b Nemotron replication as higher-priority
work.

## M33 Authorization

M33 is stable-core plus EMA dynamic recalibration. It combines:

- M26's per-model stable core `C`;
- total top-5 protection budget matching M11b's passing configuration;
- an allocation fraction `f in {0.25, 0.5, 0.75}` reserved for stable-core
  channels, with the remaining budget assigned by EMA-smoothed dynamic
  selection at `alpha=0.3`.

M33 is **not** authorized immediately. It runs only after M11b replication on
Nemotron-3-Nano:

- If M11b Nemotron shows at least partial signal, M33 is authorized.
- If M11b Nemotron kills entirely, M33 is unlikely to help and should remain
  deferred.
- If M11b Nemotron is ambiguous, use judgment: run M33 on Granite-Small first
  if the Granite evidence still indicates both component mechanisms carry
  signal.

If authorized, M33 has higher expected information value than M16 because it
directly combines the two landed partial-positive signals.

## M16 Authorization

M16 is dwell-time filtered protection. It differs from M2/M10/M11/M33 because
it filters eligibility: a channel must remain in the top-1% for `N`
consecutive positions before it qualifies for protection. Test values are
`N in {3, 5, 10}`.

M16 runs after M33:

- If M33 passes or is ambiguous with real signal, run M16 to test whether
  dwell-time filtering adds a distinct mechanism.
- If M33 kills cleanly, M16 still runs because dwell-time filtering is
  structurally different from stable-core plus dynamic EMA.
- If GPU budget becomes tight, drop M16 before dropping M33.

## Explicit Non-Authorizations

The following are not authorized in the current sprint:

- M17 eviction cooldown: too redundant with M11 EMA and M16 dwell-time
  filtering.
- M27 layer-stratified protection: demoted and no longer prioritized because
  per-component dissection found comparable drift rates across SSM, attention,
  and MoE/block-output categories.
- Streaming-PCA / online subspace tracking: the FFT analysis found broadband
  spectra, making slowly varying subspace methods low expected value.
- M11 plus M26 ensemble voting: M33 is the cleaner combination test.
- M28 long-decode calibration as a standalone method: potentially useful as a
  modification to existing methods, but not authorized as a separate sprint
  method.

## Synthetic-Validation Caveat

Synthetic checks for M33 and M16 were intentionally treated as mechanism
sanity checks only. They were too easy and saturated near `1.0` recovery under
simplified assumptions. They do not validate real-world performance because
real activation channels are correlated, heavy-tailed, task dependent, and
their errors propagate nonlinearly through attention, SSM state, and KV cache.

## Queue Implication

After KL accumulation and ParoQuant Granite-Small complete, the revised
priority is:

1. M11b replication on Nemotron-3-Nano.
2. M33 if M11b Nemotron shows partial or stronger signal.
3. M16 after M33 according to the conditional sequencing above.
4. M11b replication on DeepSeek-R1-Distill-Qwen-1.5B if budget allows.
5. M27 only if the higher-priority items complete and budget remains.

## What Would Invalidate This Decision

This decision should be revisited if M11b replication on Nemotron kills
unambiguously, if KL accumulation shows a more urgent compound-error mechanism
that prioritizes M31, or if the human decides that the wide-CI M11b Granite
PASS is sufficient to skip combination methods and move directly to paper
integration.
