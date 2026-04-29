# Repo Readiness Review

- date: `2026-04-26`
- status: scoped positive-method manuscript with strengthened same-family/remap
  packet evidence; not yet a full cross-family ICLR claim
- estimated distance: one successful bidirectional cross-family method gate or a
  narrower paper framing that explicitly limits the claim to same-family/source-
  private packet communication

## Current Paper Story

The honest current story is conditional innovation rather than proven latent
transfer. Target-side candidate/self-repair gives a strong decoder floor, C2C
shows real cache-level headroom, and LatentWire has a mature evaluation harness.
What is missing is a deployable LatentWire method whose improvement is both
source-derived and stable under source-destroying controls.

Update `2026-04-29`: the current strongest contribution set is now (1)
source-private scalar packets, (2) strict source-destroying controls, (3)
byte/latency systems frontiers, (4) QJL/TurboQuant-style matched-byte residual
comparator, and (5) canonical RASP relative-score packets. Canonical RASP
serializes candidate scores by stable public candidate identity instead of
display order and passes a larger frozen worst-remap slice (`0.442` vs `0.361`
scalar and `0.250` target, controls clean), but the seven-remap bootstrap still
misses strict pass by a small CI margin (`+0.146` vs `+0.150`) and bidirectional
cross-family remains failed. The live next branch is consistency-distilled
canonical posterior packets; the exact blocker to full-paper readiness is
bidirectional cross-family evidence or a defensible decision to frame the paper
as same-family/protocol-assisted source-private communication with explicit
cross-family limitations.

Update `2026-04-29`: the JEPA/consistency-inspired
`consistent_posterior_packet` was implemented and pruned as a cross-family fix.
It improved the failed core-to-holdout direction on the medium slice (`0.381`
vs `0.250` target), but the larger slice failed controls: source `0.354`,
scalar `0.370`, and order-mismatch `0.355`. Holdout-to-core remained positive
(`0.495` vs `0.250`) but did not beat canonical RASP (`0.502`). This result
weakens the hypothesis that more posterior smoothing alone solves
cross-family. The highest-priority full-paper strengthening gate is now a
systems-rate frontier with TTFT/latency, source generation cost, decode cost,
bytes/tokens, and matched structured-text/compression baselines. Cross-family
should be treated as an explicit limitation until a new surface or architecture
passes bidirectional controls.

Update `2026-04-29`: a deterministic rate-frontier artifact now strengthens the
systems claim. On frozen core and holdout reviewer-risk surfaces, the
source-private packet reaches oracle accuracy at `2` bytes; JSON/free-text
structured relays require at least `21`/`17` bytes to reach oracle, and full
hidden-log relay is `183.2x-186.7x` larger. Matched-byte text at the packet
rate remains at target-only accuracy. This supports a clean far-left byte-rate
frontier contribution. Remaining systems gap: this is not TTFT or server
throughput, so endpoint telemetry is still needed before making serving
latency claims.

Update `2026-04-29`: the CPU systems frontier aggregate now gives the paper one
reproducible table with wins, near misses, and failures. It includes `32` rows:
rate frontier, 5-seed scalar packet stability, remapped codebooks, canonical
RASP, model-emitted packets on current small local models, target-model decoder
rows, bidirectional cross-family falsification, and the pruned
consistency-posterior ablation. This strengthens the systems/robustness
contribution and reduces cherry-picking risk, but it does not change readiness:
the paper remains a scoped positive-method manuscript rather than a full
cross-family ICLR-ready claim. The exact blocker is now sharper: larger
target-decoder replication and diagnostic-code remap/paraphrase stress testing
are needed before reviewers will accept the receiver and protocol as general
rather than hand-coded.

Update `2026-04-29`: the protocol-stress aggregate sharpens the uniqueness
claim and the remaining reviewer risk. Deterministic diagnostic-codebook remaps
pass across three `500`-example codebooks and four byte budgets; learned
slot-feature remaps pass at `6` bytes with weaker margins; canonical
candidate-order RASP rows remain positive but are marked as near misses because
their aggregate bootstrap does not clear the strict CI threshold. The paper now
has a clearer technical-contribution stack: source-private evidence-packet
benchmark/control protocol, compact scalar packet method, byte-rate systems
frontier, QJL/TurboQuant-style matched-byte comparator, canonical RASP
candidate-relative packet, and protocol-stress/uniqueness accounting. It still
needs either query-aware compressed-text baselines or learned target-decoder
prompt-paraphrase stress before claiming the protocol objection is mostly
handled.

Update `2026-04-29`: the rate frontier now includes a query-aware compressed
text baseline, not only naive JSON/free-text truncation. The query-aware row
extracts the shortest diagnostic-span text form from the private log and reaches
oracle at `14` bytes on both frozen surfaces. The packet still reaches oracle at
`2` bytes, giving a `7.0x` byte advantage over this stronger compressed-text
baseline and `10.5x` over JSON. This improves the systems contribution, but
does not create endpoint TTFT evidence or solve the learned-receiver objection.

Update `2026-04-29`: the learned Wyner-Ziv packet gate is now the strongest
answer to the “hand-coded deterministic code” objection. It passes `9/9`
remapped slot-codebook rows across `2/4/6` byte budgets, with scalar WZ accuracy
`0.418-0.508` versus target-only `0.250` and clean source-destroying controls.
This adds a real learned source-private syndrome contribution. It does not
change the full-paper blocker: bidirectional cross-family evidence and a larger
target-model decoder are still missing.

Update `2026-04-29`: bidirectional cross-family learned WZ fails. The
`core_to_holdout` direction is below target at every budget and is explained by
source-destroying controls; `holdout_to_core` has a strong 6-byte row but does
not establish symmetry. This confirms the paper must not claim general
cross-family latent transfer. The strongest defensible full-paper framing is now
a source-private packet benchmark/method paper with learned same-family/remap
packets, byte-rate systems wins, and explicit cross-family negative evidence.

Update `2026-04-29`: a protected rotated residual packet ablation now addresses
the TurboQuant/QJL-style codec objection but does not become a new headline
method. It is source-control positive on all `9/9` remap/budget rows and
slightly improves the 2-byte scalar WZ row on two remaps, but strict promotion
fails because p50 decode latency is `3.56-7.33 ms` rather than `<2 ms`, and two
6-byte rows trail scalar WZ by more than `0.02`. Keep this as a principled
systems/compression comparator. The next full-paper strengthening gate should
remain `n=256` target-model decoder replication or a genuinely new
anchor-relative/dictionary packet for cross-family.

Update `2026-04-29`: the target-model receiver branch now has a progress-enabled
harness and a positive `n=16` core/holdout smoke. Frozen Qwen3-0.6B decodes the
matched 2-byte packet at `0.688` core and `0.750` holdout while target-only and
shuffled-packet controls remain `0.250`, with valid prediction rate `1.000`.
This weakens the hand-coded-decoder objection but does not close it. The next
readiness-critical gate is `n=64` or `n=160` with all six receiver controls.

Update `2026-04-29`: the frozen target-decoder receiver now has strict-small
all-control evidence. Qwen3-0.6B CPU passes at `n=32` on both core and holdout:
core matched `0.688` versus target/best control `0.250`; holdout matched
`0.750` versus target `0.250` and best control `0.281`. Exact-ID parity and
valid prediction rate are `1.000`. This materially weakens the hand-coded
decoder objection, but does not make the paper fully ICLR-ready: the receiver
still needs `n=160`/`n=256` replication or endpoint TTFT telemetry, and
bidirectional cross-family transfer remains failed. Literature synthesis
confirms the safe novelty claim is source-private extreme-rate evidence
handoff, not generic latent communication. The top new method branch is now
anchor-relative sparse innovation packets to target the scalar WZ/canonical
RASP cross-family failure.

Update `2026-04-29`: the static anchor-relative sparse innovation packet branch
has been tested and pruned as a bidirectional cross-family fix. AR-SIP passes
holdout-to-core at some budgets (`0.496` at 2 bytes and `0.373` at 8 bytes with
clean controls), but core-to-holdout stays at or below target-only and
anchor/random controls can dominate. This is a serious negative for shallow
relative-coordinate fixes. It strengthens the paper's honesty and narrows the
next branch: either add endpoint TTFT/E2E systems telemetry for the existing
positive packet, or pursue a learned target-preserving query bottleneck /
Q-Former-style receiver rather than another static sparse code.

Update `2026-04-29`: the first endpoint-proxy systems row is now positive. A
Qwen3-0.6B CPU receiver passes on core and holdout at `n=8` and `n=16` with
endpoint-style prompt/token/TTFT/E2E logging. The stronger `n=16` rows reach
matched 2-byte packet accuracy `0.688` on both surfaces versus target-only and
matched-byte text at `0.250`; query-aware text is `7.0x` larger and full
hidden-log relay is `183.2x-186.7x` larger. Full-log p50 TTFT is `+165.4 ms` to
`+190.7 ms` relative to the packet at `n=16`. This materially improves the
systems story, but the readiness status remains scoped positive-method rather
than ICLR-full ready: the endpoint evidence is still `n=16 + n=16` local CPU
proxy, not server throughput, and cross-family latent transfer remains failed.
The next gate is `n=64`/`n=160` endpoint-proxy replication with prompt
paraphrase stress, then a vLLM/OpenAI-compatible server run when GPUs are
available.

Update `2026-04-29`: endpoint prompt-paraphrase stress now gives a cleaner
receiver-boundary story. I added `canonical`, `terse`, and `audit` prompt
styles to the endpoint-proxy harness. The `audit` paraphrase passes at `n=32`
on both frozen surfaces: core matched packet `0.719` versus target `0.250` and
matched-byte text `0.281`; holdout matched packet `0.844` versus target and
matched-byte text `0.312`. Full-log relay is still `183.2x-186.7x` larger and
adds `+157.4 ms` to `+163.4 ms` p50 TTFT versus the packet. The deliberately
under-specified `terse` prompt fails on core `n=16`, with packet `0.250`, equal
to target. This weakens the one-prompt-artifact objection, but also makes the
limitation explicit: the endpoint receiver needs a clear public side-
information contract. Next harness fix before promotion: split strict label
emission from diagnostic-code-mapped accuracy and add a deranged candidate-
diagnostic table control.

Update `2026-04-29`: the endpoint harness now includes the strict-label /
diagnostic-mapped split and two endpoint source-destroying controls:
`random_same_byte_packet` and `deranged_candidate_diag_table`. The audit prompt
passes both surfaces at `n=16` under the stricter gate. Core: matched packet
`0.750`, target `0.250`, best source-destroying control `0.250`, random same-
byte `0.000`, deranged public table `0.000`. Holdout: matched packet `0.875`,
target `0.312`, best source-destroying control `0.312`, random same-byte
`0.125`, deranged public table `0.000`. The important caveat is strict label
emission: packet strict-label accuracy is only `0.062` core and `0.250`
holdout, so the endpoint evidence supports protocol-code decoding via public
side information, not free-form candidate-label generation. The next scale gate
is `n=64` canonical+audit with strict controls enabled.

Follow-up `2026-04-29`: the same strict-control audit endpoint gate now passes
at `n=32` on both frozen surfaces. Core: matched packet `0.719`, target
`0.250`, best source-destroying control `0.281`, random same-byte `0.031`,
deranged public table `0.000`, and full-log p50 TTFT `+159.2 ms` versus the
packet. Holdout: matched packet `0.844`, target `0.312`, best
source-destroying control `0.312`, random same-byte `0.094`, deranged public
table `0.000`, and full-log p50 TTFT `+185.8 ms`. Strict label-only packet
accuracy remains low (`0.156` core, `0.219` holdout), so the claim remains
protocol-code decoding with public side information. The CPU systems frontier
now has `84` rows. The next exact gate remains `n=64` canonical+audit strict
controls; if that passes, widen to `n=160` and then run a server-side
TTFT/throughput benchmark when NVIDIA GPUs are available.

Follow-up `2026-04-29`: payload-gated rescoring fixed a parser loophole where
the endpoint receiver could hallucinate an untransmitted diagnostic code and
still receive diagnostic-mapped credit. Under the stricter parser, the audit
rows remain accuracy-positive but fail the valid-output gate; core `n=64`
audit is a near miss with packet `0.750`, target `0.250`, best
source-destroying control `0.203`, and full-log p50 TTFT `+260.2 ms`, but
packet valid rate is only `0.781`. I added a `label_strict` receiver prompt
that requires exact candidate-label output. It passes both frozen surfaces at
`n=16`: core packet `0.688` versus target/control `0.250`, holdout packet
`0.625` versus target/control `0.250`, and packet valid rate `1.000` on both.
This becomes the live endpoint receiver branch. The CPU systems frontier now
has `87` rows. Next exact gate: label-strict `n=32` core+holdout, then `n=64`
if both pass.

Follow-up `2026-04-29`: label-strict `n=32` passes on both frozen endpoint
surfaces. Core: packet `0.688`, target `0.250`, matched-byte text `0.250`,
random same-byte `0.000`, deranged public table `0.219`, valid rate `1.000`,
strict-label packet accuracy `0.656`, and full-log p50 TTFT `+164.8 ms`.
Holdout: packet `0.656`, target `0.250`, matched-byte text `0.250`, random
same-byte `0.000`, deranged public table `0.250`, valid rate `1.000`,
strict-label packet accuracy `0.625`, and full-log p50 TTFT `+167.1 ms`.
This materially improves the endpoint contribution: the live receiver now
outputs exact candidate labels at n32, not just protocol codes. The CPU systems
frontier has `89` rows. Next exact gate: label-strict `n=64` core+holdout.

Follow-up `2026-04-29`: label-strict `n=64` passes on both frozen endpoint
surfaces. Core: packet `0.703`, target `0.250`, matched-byte text `0.250`,
random same-byte `0.000`, deranged public table `0.234`, valid rate `1.000`,
strict-label packet accuracy `0.672`, and full-log p50 TTFT `+217.2 ms`.
Holdout: packet `0.672`, target `0.250`, matched-byte text `0.250`, random
same-byte `0.000`, deranged public table `0.250`, valid rate `1.000`,
strict-label packet accuracy `0.656`, and full-log p50 TTFT `+192.7 ms`.
This clears the local n64 endpoint receiver gate and materially strengthens
the full-paper case. The CPU systems frontier has `91` rows. Next exact gate:
paired uncertainty on the n64 rows, then frozen label-strict `n=160`
core+holdout.

Update `2026-04-27`: the no-harm CPU replay kills shallow source-predicate
decoding on current artifacts. A 4-bit candidate syndrome still has source
specificity on holdout (`4` clean source-necessary IDs, control clean union
`0`) but harms `14` target-self examples. Stronger abstention removes harms but
also removes clean gains. The live branch remains `none`; next viable branches
are learned semantic predicates with erasure-aware abstention and zero-init
target-preserving query bottlenecks after the MPS blocker clears. The blocker
persisted after both `kill -9` and `sudo kill -9`, so the next action is
OS/session-level cleanup before any MPS experiment.

Update `2026-04-27`: the 12-row 7B disagreement answer-likelihood smoke kills
normalized-answer receiver-likelihood variants on that surface. The harness now
has explicit `answer_only` and `answer_masked_source` candidate-pool controls
plus score-matrix collapse telemetry. Matched and answer-only sketches are
byte-identical, answer-masked-source recovers no clean IDs, and the gate reports
`0` clean source-necessary IDs after controls. This branch is pruned unless a
new surface first shows answer-unexplained target-pool headroom. The live branch
remains `none`; the top next move is upstream source-surface discovery or a
JEPA-style answer-masked trace/latent objective only after such headroom exists.

Update `2026-04-27`: after MPS cleared, a cached
`Qwen/Qwen2.5-7B-Instruct -> Qwen/Qwen3-0.6B` SVAMP70 scout produced stronger
raw source headroom but still failed the answer-masked promotion gate. Target
was `21/70`, source `15/70`, text relay `12/70`, clean source-only after text
relay `7`, and target/source oracle `29/70`; however only `3` clean IDs were in
the target-side pool and all were source-final/verified-answer explained
(`answer_unexplained_clean_in_pool = 0`). A CPU answer-free
query-bottleneck syndrome probe on SVAMP32 also failed: matched `10/32`,
target-only `14/32`, clean source-necessary `0`. The live branch remains
`none`; do not spend on another receiver/connector unless a Math-7B or selected
fresh surface first exposes answer-unexplained target-pool headroom.

Update `2026-04-27`: the selected-disagreement
`Qwen/Qwen2.5-Math-7B-Instruct -> Qwen/Qwen3-0.6B` scout is now a sharper
negative. On the 12-ID disagreement slice, target was `0/12`, source `5/12`,
and text relay `1/12`, with exact ID parity and full numeric coverage. The
answer-masking audit found `5` clean source-only IDs, `3` clean IDs in the
target-side pool, and `answer_unexplained_clean_in_pool = 0`. The model is now
cached locally after retrying with `HF_HUB_DISABLE_XET=1`, so a full SVAMP70
Math-7B scout is executable, but the live branch remains `none`. JEPA/LeJEPA/
V-JEPA guidance is still anti-collapse harness design only: answer-masked dual
views, frozen target latent/KV targets, matched-source margins over controls,
target preservation, and variance/effective-rank/covariance telemetry.

Update `2026-04-27`: the full SVAMP70 Math-7B scout fails the source-surface
gate. Target is `21/70`, source `5/70`, text relay `8/70`, clean source-only
after text relay is `3`, clean in target-side pool is `1`, and
`answer_unexplained_clean_in_pool = 0`. A follow-up target-only sampling smoke
on the three residual clean IDs raises combined no-source target-pool
reachability to `2/3`, but the source selector is pruned: `full` and
`answer_only` sidecars both recover `2/3`, while `answer_masked` recovers
`0/3`. The live branch is no longer source-surface/receiver tuning; it is
target-only/no-source candidate-pool generation followed by strictly
answer-masked source selection controls.

Update `2026-04-27`: the SVAMP32 source-sampling smoke creates a new discovery
surface but not a positive method. `Qwen/Qwen2.5-Math-1.5B` source-reasoning
sampling reaches only `10/32` oracle versus the target/no-source full32 S8 pool
at `14/32`, so there is no accuracy claim. It does, however, add two C2C-clean
residual IDs beyond the target/no-source pool (`6e9745b37ab6fc45`,
`de1bf4d142544e5b`). The live branch is now a strict source-conditioned
selector or JEPA-style rate-capped connector on those two IDs. Promotion
requires matched-only recovery, control clean union `0`, no target-correct harm,
and bytes plus collapse telemetry.

Update `2026-04-27`: the two-ID source-sampled branch is pruned as a
source-specific connector surface. A strict candidate selector over the appended
source-sampled candidates fails under full, answer-only, and answer-masked
source profiles (`0` matched clean recoveries, `5-6` accepted harms). A 16-sample
replay keeps matched source at `2/2`, but the target model with the same
brief-analysis wrapper also reaches `2/2`; target direct remains `0/2`, and
source direct is `1/2`. The live branch is no longer the two-ID source-sampled
surface. The prompt wrapper must become a target-prior baseline/control before
any future source-communication claim.

Update `2026-04-27`: full SVAMP32 target brief-wrapper sampling is now promoted
to a mandatory target-prior baseline. Target brief-wrapper S4 reaches `18/32`
oracle and `4/6` C2C-clean residual IDs. The union of target direct S8 and
target brief-wrapper S4 reaches `23/32` oracle and all `6/6` C2C-clean residual
IDs. Adding the source brief S4 surface raises total oracle only to `24/32` and
adds `0` new C2C-clean residual IDs. The source-sampling family is pruned as a
communication surface; the next source-surface discovery must subtract target
brief-wrapper priors at matched or larger budget.

Update `2026-04-27`: prompt-wrapper controls now partially prune the best
reusable larger source surfaces. On the Math-7B SVAMP70 clean7 surface, target
brief-wrapper S8 recovers `4/7` clean source-only IDs without source input,
leaving only three residual candidates (`33836927fc9f1a8a`,
`4c84ebf42812703b`, `d64f6e35083ffe8c`) for source-destroying controls. On GSM
clean2, source brief S8, target direct S16, and target brief S16 all recover the
same single ID, so source adds `0` IDs beyond the target prompt union. The live
branch remains prompt-controlled source-surface discovery; JEPA-style connectors
remain deferred until a target-prior-unexplained surface survives answer-masked
controls.

Update `2026-04-27`: the stricter SVAMP32 clean C2C residual target-only
sampling gate separates generator headroom from communication. No-source target
sampling reaches gold on `2/6` clean residual IDs with full numeric coverage,
but the source selector fails: `full` and `answer_only` both select `0/6`
correct clean IDs, and `answer_masked` accepts nothing. This preserves
target/no-source candidate-pool generation as reusable headroom, but kills the
current numeric source-candidate sidecar as a positive method. The next live
candidate is a JEPA-style answer-masked process/latent ranking smoke on the two
reachable clean IDs with frozen target/candidate latents, source-destroying
controls, no-harm accounting, and collapse telemetry.

Update `2026-04-27`: the first answer-masked process-trace ranking smoke also
fails. A deterministic TF-IDF/process-text sidecar over the SVAMP32 clean6
target-only pool gives matched clean correct `0/6` across all variants. The
best diagnostic variant, prediction-only with `t2t` excluded, accepts two
matched IDs but both are wrong, while a random-sidecar control recovers the only
clean correct ID. Collapse telemetry is healthy enough (`effective_rank
31.27`, zero vectors `0`), so this is not low-rank representation collapse; it
is lack of source-necessary process signal and residual source-number leakage
(`5/6` selected values appear among unmasked source numbers). Hand-built
numeric and process sidecars are now pruned on this slice.

Update `2026-04-27`: prompt-wrapper source-surface controls further narrow the
search. On Math-7B SVAMP70 clean7, target brief-wrapper S8 reaches `4/7` clean
source-only IDs, leaving only three residual discovery IDs and no immediate
answer-unexplained candidate-pool signal. On GSM70 clean2, source brief S8,
target direct S16, and target brief-wrapper S16 all reach the same single ID,
so source adds `0` residual IDs beyond the target prompt union. The live branch
remains source-surface discovery only; JEPA/LeJEPA/V-JEPA-style connectors stay
deferred until a surface has at least three target-prior-unexplained residual
IDs under source-destroying controls.

Update `2026-04-27`: the deadline plan is pivoting from ordinary math residual
hunting to source-private residual communication. The new top paper story is a
rate-capped source message decoded with target side information: private
evidence packets, private tool-trace distillation, candidate-syndrome sidecars,
and only then Query-JEPA/adapter connectors. The strict-small pass bar is now a
source-private benchmark where target wrapper `S32` and no-source oracle fail,
matched source sidecar beats best no-source by at least `15` points, zero/
shuffled/random/answer-only controls stay near no-source, and structured text at
matched bytes is included. This is a method/benchmark reset, not positive
evidence yet.

Update `2026-04-27 00:50 PDT`: KVComm is now harness-ready for strict
source-control evaluation, but it remains baseline/tooling evidence rather than
a positive method. The wrapper supports matched, zero-source, shuffled-source,
and target-only final modes under one matched-only layer selection. A CPU smoke
over two examples verifies provenance and fixed-layer reuse, with all modes at
`0/2` and shuffled-source using nonmatching source IDs. The live branch remains
`none`; the highest-priority executable gate after MPS clears is the
`kvcomm_svamp32_controls_smoke_20260427` command recorded in the ledger.

Follow-up hardening replaced fixed-offset shuffled-source with deterministic
hash-based non-self pairing and answer-overlap logging; the CPU smoke still
passes as tooling-only evidence.
Second follow-up hardening added configured paired sidecar baselines so KVComm
artifacts now report matched-vs-target-only, matched-vs-zero-source, and
matched-vs-shuffled-source flip tables.
Third follow-up hardening added cache-derived byte telemetry. On the CPU smoke,
selected-layer KVComm controls average `530432` communicated bytes/example,
while target-only is `0`; this is systems telemetry only, not method evidence.

Update `2026-04-28`: the source-private candidate-syndrome protocol cleared a
deterministic strict-small gate (`160` examples, matched `1.000`, best
no-source/control `0.250`, `2` byte packet), but the first model-produced packet
smoke falsifies the naive cryptographic-source-agent branch. On a frozen
`16`-example smoke with `Qwen/Qwen2.5-0.5B-Instruct`, matched model packets stay
at target-only (`4/16`, `0.250`) while source-final-only reaches `16/16`.
Generated packets mostly copy instruction/key/record fragments instead of
computing the digest. Current readiness remains not ICLR-ready. The live branch
is now source-private handoff with naturally emitted private evidence packets;
the next exact gate is `source_private_testlog_packet_strict_small_20260428`.

Update `2026-04-28`: the source-private test-log packet branch cleared a
deterministic strict-small gate. On `160` examples, a two-byte private
`TRACE_SIG` packet gives matched `160/160` while target-only, target-wrapper,
zero, shuffled, random, answer-only, answer-masked, target-derived, and
matched-byte raw-log controls stay at `40/160`. Full log and full signature
oracles are `160/160`, public target prompts contain no private log or
`TRACE_SIG`, and matched packets copy no candidate/answer labels. This promotes
source-private test-log handoff as the live branch, but readiness remains not
ICLR-ready because packet generation and decoding are still deterministic. The
next exact gate is `source_private_testlog_packet_llm_packet_20260428`.

Update `2026-04-28`: the same source-private test-log surface now has a
positive model-mediated strict-small result. With
`Qwen/Qwen2.5-0.5B-Instruct` as source packet extractor, matched model packets
reach `150/160` (`0.938`) versus target-only and all source-destroying controls
at `40/160` (`0.250`); full signature oracle is `160/160`, packet valid rate is
`0.91875`, and p50 matched source-packet latency is `162.51` ms. This is the
current live positive branch. Readiness remains not ICLR-ready because it is one
model/seed on a synthetic signature benchmark with source-side private-log line
isolation and a deterministic decoder. Next gate: seed repeat with a
no-helper-line ablation and then a hidden-test/code-repair variant.

Update `2026-04-28`: seed-repeat and prompt-mode ablation narrow the live claim.
Helper-line protocol runs with seeds `29` and `30` both pass at `150/160`
matched accuracy versus `40/160` target/control floor. The no-helper full-log
ablation fails at `55/160` with low valid-packet rate (`0.163`). This keeps the
branch alive as protocol-assisted private tool-log packet emission, but it is
not a general unstructured-log extraction result. Current readiness remains not
ICLR-ready. Next gate: cross-model/source-family confirmation on the same frozen
IDs, then a hidden-test/code-repair variant.

Update `2026-04-28`: cross-model helper-line confirmation passed on capable
instruction-tuned source models. `Qwen/Qwen2.5-0.5B-Instruct` reaches `150/160`,
`Qwen/Qwen3-0.6B` reaches `160/160`, and `microsoft/Phi-3-mini-4k-instruct`
reaches `146/160`, all against a `40/160` target/control floor. TinyLlama fails
at `40/160` with `0` valid packets, so it is a negative capability row. This
promotes the branch as cross-model protocol-assisted private tool-log packet
handoff, not universal extraction. Current readiness remains not ICLR-ready
until a hidden-test/code-repair variant replaces synthetic signature fields.

The strongest bound is the SVAMP32 C2C-derived syndrome sidecar:

- strict target-side pool: `14/32`
- clean source-necessary IDs: `2/6`
- controls clean union: `0/6`
- syndrome size: `1` byte
- blocker: the syndrome uses C2C numeric answers, so this is a bound, not a
  deployable positive method

The first C2C-mechanism distillation attempt is now negative:

- scalar prefill trace: matched `11/32`, clean source-necessary `0/6`
- residual prefill trace: matched `12/32`, clean source-necessary `0/6`
- target-only decoder floor: `14/32`
- decision: do not scale summary-feature C2C syndrome distillation without a
  new token/layer-level mechanism reason

The first Perceiver answer-teacher contrastive connector is also negative:

- fixed gates `0.125`, `0.15`, `0.20`: matched-only clean residual IDs `0/6`
- matched-positive clean IDs: `2/6`, but both are explained by shuffled-source,
  target-only, or slots-only controls
- decision: do not run generation for this checkpoint

The same branch also fails on the stronger SVAMP70 C2C-vs-process-repair
surface:

- clean C2C source-only IDs after excluding process-repair: `10`
- teacher-forced gate `0.15`: matched-positive clean `4/10`
- matched-only clean: `0/10`
- control-leak clean: `4/10`
- decision: kill this connector family until the objective changes

The first objective-level rescue also fails on SVAMP32:

- added training-time anti-memory controls against `target_only` and
  `slots_only`
- fixed gates `0.125`, `0.15`, `0.20`: matched-only clean residual IDs `0/6`
- matched-positive clean IDs: `2/6`, but both are explained by zero-source or
  slots-only controls
- mean matched-control delta remains negative at all tested gates
- decision: do not run generation; pivot away from receiver-conditioned
  Perceiver/delta-memory signal formation unless a materially new objective or
  architecture reason appears

The first true source-conditioned soft-prefix logprob gate is also negative:

- source-only matched connector with fold-local feature standardization,
  numeric-only distractors, and mean-token continuation logprob
- matched-only clean source-communication candidate IDs: `1/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.771126`
- decision: kill global summary soft-prefix connectors on this surface before
  generation

The first token-local cross-attention rescue also fails its first rung:

- target-query cross-attention into standardized source token states
- matched-only clean source-communication candidate IDs: `0/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.383649`
- decision: do not scale this exact tiny prefix-emitting cross-attention
  connector by epochs or width without a new hypothesis

The source-control contrastive variant of that cross-attention gate also fails:

- training penalizes zero-source, shuffled-source, same-norm-noise, and
  projected-source controls when they match or beat the real-source margin
- matched-only clean source-communication candidate IDs: `0/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.382854`
- decision: objective-level control penalties do not rescue this tiny
  prefix-emitting cross-attention architecture; do not tune this exact family
  further without a larger architectural change

The target-side continuation-loss rescue of the same family also fails:

- training objective changed from gold-vs-distractor margin to target
  continuation next-token CE
- heldout logprob on SVAMP32 clean C2C-headroom IDs: matched-only clean `0/6`,
  clean control leaks `4/6`, mean matched-minus-control clean margin
  `-0.194783`
- 64-token generation on the six clean IDs: matched `1/6`, while zero-source,
  shuffled-source, target-only prefix, and slots-only prefix each reach `2/6`
- decision: kill the low-capacity prefix-emitter family on this surface; the
  next learned-interface branch must expose a larger, rate-controlled source
  memory or use a different source/surface pair

The current live branch is now the query-innovation/source-memory resampler:

- historical GSM8K32 query-innovation rows are finite and target-safe enough to
  probe, but their small accuracy gains were retained under zero/shuffled
  source controls
- this cycle added eval-only gold-answer continuation scoring to
  `latent_bridge/evaluate.py`, so the next gate can compare matched source
  against zero-source, shuffled-source, target-only, and slots-only controls on
  the same generated-answer surface
- CPU micro-smoke on four GSM8K examples fails: matched mean answer logprob
  `-7.025400`, zero-source `-6.925437`, shuffled-source `-7.048394`,
  slots-only `-7.025400`, matched-best-control delta `-0.115530`, and
  matched best-control wins/losses/ties `0/4/0`
- decision: kill the current finite non-target-conditioned query-innovation
  checkpoint as a live source-communication row
- next branch: target-conditioned query-innovation/source-memory connector
  that supports `target_only` and `slots_only` controls from the first gate
- current blocker: an orphaned MPS calibration process, PID `31103`, is stuck
  in `STAT=UE` and ignores `SIGKILL`; do not start more MPS runs until it is
  cleared

The target-conditioned query-memory follow-up is now also negative:

- SVAMP32 delta-memory CPU answer-likelihood smoke fails because matched source
  loses to zero-source on mean answer likelihood and wins `0/4` against the
  best runnable control
- SVAMP70 Perceiver answer-teacher CPU answer-likelihood smoke fails with
  best-control wins `0/4` and mean matched-minus-best-control delta
  `-0.112360`
- Qwen2.5-Math SVAMP32 Perceiver has one partial positive clue on four clean
  IDs: matched beats every control with best-control wins `3/4` and mean
  live-best delta `+0.080362`
- the required six-clean-ID expansion fails: matched still beats zero-source,
  but loses on mean to shuffled-source, target-only, and slots-only controls;
  mean matched-minus-best-control delta is `-0.090384`
- decision: kill target-memory/query-memory Perceiver checkpoints as the
  current live positive-method branch; no method is live until a source
  surface/interface reset selects the next branch
- current blocker remains PID `31103`, the stuck MPS calibration process

The subsequent no-harm source-predicate replay is also negative:

- candidate syndrome bits4: live clean source-necessary `1` with `16`
  target-self harms; holdout clean source-necessary `4` with `14` target-self
  harms; control clean union `0`
- source predicate router with stronger no-harm pressure: best rows reach only
  `23/70`, clean `3`, accepted harm `1`, and fail the matched-correct gate
- source likelihood no-harm gate: accepted harm `0`, control clean union `0`,
  but clean source-necessary `0` on both live and holdout
- decision: prune shallow numeric/hash syndrome and source-text predicate
  routers on current artifacts; revive only learned semantic predicates with
  erasure-aware abstention or stronger source surfaces

The learned semantic-predicate CPU decoder is now also negative on holdout:

- new analyzer: `scripts/analyze_svamp_source_semantic_predicate_decoder.py`
- strict harm20 gate: live `25/70`, clean source-necessary `3`, accepted harm
  `0`, control clean union `0`
- holdout: `9/70`, clean source-necessary `0`, accepted harm `0`, control
  clean union `0`
- decision: target-safe live recovery is possible, but it does not transfer;
  prune generated-source-trace semantic predicate decoding on current
  Qwen2.5-Math -> Qwen3 SVAMP artifacts

The CPU target-likelihood receiver follow-up is also negative on live:

- scorer: `Qwen/Qwen3-0.6B` on CPU over target/text/source normalized answer
  candidates
- target-alone/text/source candidate correctness: `21/70`, `22/70`, `13/70`
- top-likelihood selection: `14/70`, with source selected on `64/70`
  examples
- accept-all source-top recovers all `6` clean live source-only IDs, but harms
  `16` target-correct examples
- simple no-harm live thresholds recover at most `1` clean source-only ID and
  remain around `22-23/70`, below the `25/70` live gate
- decision: prune this target-likelihood receiver variant before holdout; a
  future receiver-gate claim needs true condition-specific rescored controls
  rather than sketch shuffling or forced target fallback

The frozen CPU source-candidate sidecar materializer is also negative:

- new materializer: `scripts/materialize_svamp_source_candidate_sidecars.py`
- emits 1-byte `candidate_scores` sidecars over target-side candidate values
  only; source-only values are not added to the receiver pool
- live materialization has source final in target pool `43/70` and
  source-mentioned target-pool hits `59/70`, but the hardened decoder remains
  at `21/70`, clean source-necessary `0`, accepted harm `0`
- holdout reaches `11/70`, accepted `7`, clean source-necessary `0`, accepted
  harm `1`

Update `2026-04-27 04:35 PDT`: the sampled clean3 target-pool sidecar briefly
passed smoke but was killed by source-answer ablation. Target-only sampling made
one remaining clean source-only ID reachable, and a full source candidate-score
sidecar selected it with controls clean-empty. However, masking source-final and
verified-answer numeric values removed the win, while a source-final-only
sidecar recovered the same ID. The live branch is no longer the clean3
candidate-score selector. The next highest-value branch is source-surface
discovery over existing artifacts, looking for target-side candidate pools that
contain source-necessary answers not explainable by direct source-final numeric
evidence; if none exist, MPS cleanup is required before richer same-family
surface generation.

Update `2026-04-27 04:45 PDT`: the source-surface answer-masking audit found no
stored surface that can support the stricter gate. Across `12` loaded surfaces
with clean IDs, every clean in-pool answer was explained by source final or
verified numeric answers; answer-unexplained clean-in-pool count was `0` for all
surfaces. Current live branch is therefore answer-masked source-interface
design and fresh surface generation after the orphaned MPS process clears.

Update `2026-04-27 04:58 PDT`: the first answer-masked process-verifier sidecar
is negative. It masks source final and verified numeric answers before
operation/equation/lexical process-overlap scoring, but live remains `21/70`
with clean `0`, and holdout remains `8/70` with clean `0`; a holdout threshold
sweep also finds no clean source-necessary recovery. The next CPU-feasible
answer-null branch is a more structured predicate syndrome, not text-overlap
process scoring.

Update `2026-04-27 05:10 PDT`: the structured answer-null predicate syndrome is
also negative. It excludes candidate values, candidate IDs, source final
numbers, verified answer numbers, and residue hashes, but live has clean `0`
with accepted harm `12`, and holdout's one clean recovery is also recovered by
random/shuffled controls. Threshold sweeps remove all clean recovery. No stored
CPU artifact now supports a useful next positive-method gate; the hard blocker
is the orphaned MPS process PID `31103`, still in `STAT=UE`, because fresh
same-family surface generation is the next required action.

Update `2026-04-27 05:32 PDT`: two fresh CPU-only SVAMP8 scouts over rows
`381..388` and `389..396` also fail as source-surface discovery. The first has
source-only `1` but clean source-only `0` after text relay and answer-masking;
the second has source-only `0`. CPU generation is feasible at about `3.3`
minutes per eight-example source/target/text scout, but brute-force CPU range
scanning is low expected value. The hard blocker remains PID `31103`; after
OS/session cleanup the next evidence-bearing gate is fresh strict-small
same-family surface generation or KVComm source-control MPS smoke.

Update `2026-04-27 05:48 PDT`: the condition-specific candidate-pool builder
is hardened. `label_shuffle_offset` is now used for a non-self donor in the
target-labeled source slot, and `shuffled_source` also guarantees non-self
donors even when offsets are zero or wrap to self. This does not revive the
killed condition-likelihood receiver, but it removes a provenance weakness
before future receiver scoring. Focused tests passed: `18` candidate/receiver/
KVComm-control tests plus `py_compile`.

Update `2026-04-27 11:10 PDT`: PID `31103` remains present after more than
`17` hours in `STAT=UE`. Added `scripts/check_mps_blocker.py` and tests so the
MPS blocker check is now executable and machine-readable:
`./venv_arm64/bin/python scripts/check_mps_blocker.py --json`. It currently
reports `"blocked": true`; no MPS work should start until it reports false.
- strict target-mentioned oracle headroom is small on the current canonical
  surface: live target `21/70` can only reach `23/70`; holdout target `8/70`
  can only reach `12/70`
- decision: kill this heuristic materializer as a method branch; keep it as
  no-leak sidecar tooling
- next branch: frozen model-scored source sidecar over target-side candidates,
  with source-destroying controls and same-byte sidecar controls from the first
  gate

The frozen model-scored target-side candidate sidecar is now also negative on
live:

- new collector: `scripts/collect_svamp_frozen_candidate_score_sidecar.py`
- scores only target-side candidate values and emits no gold/correctness fields
  or source-only values
- two-example CPU plumbing smoke passed schema/ID-parity, but accepted `0`
  sidecar rows and is not scientific evidence
- full live CPU collection over `70` examples took `351.12s`; top labels were
  `target=44`, `t2t=26`
- hardened decoder with the live sidecar: `21/70`, accepted `1`, clean
  source-necessary `0`, accepted harm `0`, control clean union `0`
- decision: kill this producer on canonical SVAMP70 live; do not spend another
  holdout pass or threshold sweep on this exact target-side candidate pool
- current next move: source-surface discovery or a qualitatively larger but
  controlled candidate surface, not another shallow sidecar over the same
  canonical SVAMP70 target pool

The prior source-only residue sidecar positive is now pruned under stricter
candidate-pool controls:

- old textless live row reached `26/70`, clean source-necessary `4/6`, but it
  included `source_alone.jsonl` as a decoder candidate artifact
- new replay excludes `source_alone` from the candidate pool and uses
  hash-based non-self shuffled-source/label-shuffle controls
- live target-side-only replay fails: best `22/70`, clean source-necessary
  `0/6`, control clean union `0`
- holdout target-side-only replay still has one clean source-necessary ID
  (`daea537474de16ac`) at `11/70`, but live failure blocks promotion
- decision: kill the old source-only residue sidecar as a paper method; treat
  the old live row as source-value candidate-pool leakage rather than clean
  communication
- next branch: source-surface discovery or a new target-side candidate surface
  that exposes enough alternatives without source-only values

Target-side candidate-pool headroom audit explains the repeated canonical live
failures:

- new audit: `scripts/analyze_target_side_candidate_headroom.py`
- canonical `svamp70_live`: target-side oracle `33/70`, but clean gold in
  target-side pool is `0/6`
- canonical holdout: target-side oracle `28/70`, clean gold in target-side pool
  `2/2`
- adjacent `chal171_240`: target-side oracle `39/70`, clean gold in pool `1/1`
- adjacent `chal241_310`: target-side oracle `23/70`, clean gold in pool `1/4`
- decision: stop tuning sidecars on canonical live until a new candidate-surface
  generator exposes target-side alternatives without source-only value leakage
- current next branch: candidate-surface generation from target self-repair,
  stochastic target routes, or non-source candidate decoders, followed by the
  target-side headroom audit before any source sidecar

The SVAMP70 exact-ID overlap audit rules out another threshold sweep on the
current canonical surface:

- canonical live has `6` clean source-only IDs, and all have been recovered by
  at least one audited branch, but the reusable recoveries cluster on a few
  live examples and come from branches that either fail holdout or harm
  target-correct cases
- canonical holdout has only `2` clean source-only IDs; only
  `daea537474de16ac` is recovered, and only by the trace-router family that
  fails the full gate
- adjacent scout positives are not canonical holdout evidence and usually come
  with target-self harm
- decision: stop CPU threshold/router sweeps on current SVAMP70 artifacts; next
  CPU work should be a true condition-specific receiver-control harness, while
  source-surface/interface reset waits for the MPS blocker to clear

The condition-specific receiver-control harness is now implemented and tested:

- new analyzer: `scripts/analyze_condition_likelihood_receiver_gate.py`
- focused tests: `tests/test_analyze_condition_likelihood_receiver_gate.py`
- verification: `12` likelihood/receiver tests passed; py_compile passed
- purpose: evaluate target-likelihood receiver gates only when each control has
  its own receiver-scored candidate pool
- status: harness-ready, not evidence; next gate is CPU collection of
  condition-specific sketches if MPS remains blocked

The condition-specific target-likelihood receiver is now killed on the current
SVAMP70 surface before control collection:

- matched-only live CV reaches only `15/70`
- clean source-necessary IDs: `1`
- accepted target-correct harm: `7`
- duplicate-answer clean IDs: `0`
- decision: do not collect remaining controls or holdout for this branch; keep
  the candidate-pool builder and duplicate-answer de-dup harness for a stronger
  source surface

The top-surface cross-attention rescue also fails:

- after consolidated surface reselection, `svamp70_live` and `svamp70_holdout`
  are the strongest source-complementary surfaces
- rerunning the same cross-attention gate on `svamp70_live` gives matched-only
  clean IDs `0/6`, clean control leaks `3/6`, and mean matched-control clean
  margin `-0.443233`
- decision: tiny learned prefix emitters are not the live branch unless a new
  mechanism directly addresses control dominance

The simplest source-only sidecar/router is also negative:

- source-generated numeric residue sidecar with target-side candidate-pool
  decoding
- source numeric coverage: `32/32`
- matched: `4/32`
- target-self preserve: `0/3`
- clean source-necessary IDs: `0/6`
- controls clean union: `0/6`
- decision: kill raw source-generated numeric residue sidecars; the clean
  issue is weak source signal, not target/control leakage

The strongest GSM mechanism clue is `dynalign_module_replace_residrank16`:

- GSM8K32 smoke: `4/32` vs target `2/32`
- GSM8K70 seed 0: `8/70` vs target `4/70`
- seed-0 source controls: zero/shuffle with target fallback retain `0/6` live
  wins
- blocker: seed stability fails; seed 3 is `2/70`, seed 4 is finite but only
  `4/70` with paired `3W/3L/64T`, and seeds 1/2 hit nonfinite checkpoint
  failures

## What Is Done

- Paper/evidence ledgers exist and are useful:
  - `paper/experiment_ledger_20260421.md`
  - `paper/benchmark_expansion_order_20260422.md`
  - `paper/reviewer_feedback.md`
  - SVAMP/GSM per-run memos under `paper/`
- Evaluation machinery is broad:
  - frozen-slice materialization and exact-ID utilities
  - paired/bootstrap uncertainty and oracle/headroom analyzers
  - source controls: zero source, shuffled source, target-only, slots-only, and
    same-norm/noise controls where applicable
  - GSM residual campaign/sweep wrappers with checkpoint health checks
  - SVAMP source/oracle/syndrome analyzers
  - C2C, KVComm, KVPress, and LatentMAS wrapper or matrix support
- Tests are healthy for unit/schema/tooling coverage:
  - `./venv_arm64/bin/python -m pytest -q`
  - result on this review: `668 passed in 26.77s`

## Saturated Or Weakened Branches

- Raw dynalign residual is a mechanism probe, not a paper method, because the
  finite repeat is target-negative and two repeat seeds are nonfinite.
- Selector-gap accept/fallback is killed as a method: fresh zero/shuffle controls
  retain gated wins, so the score is not source-specific.
- Further simple whitening/conditioning is low priority: it can prevent nonfinite
  failures but trades away the seed-0 ceiling.
- Query-pool and ID-weighted SVAMP variants are saturated below the clean gate:
  best rows recover only `1/6` clean residual IDs.
- Perceiver-query, delta-memory, contrastive delta-memory, answer-teacher
  microfit, pooled source-hidden syndrome, and learned source-token syndrome
  probes all fail the source-derived clean gate.
- C2C prefill scalar/residual summary syndrome probes fail the strict SVAMP32
  gate and remain below the target-only decoder floor.
- Perceiver-query answer-teacher plus source-control contrast fails the
  teacher-forced pre-generation gate; target/control memory still explains the
  apparent clean-ID signal.
- Scaling the same Perceiver answer-teacher connector to SVAMP70 also fails
  the teacher-forced pre-generation gate.
- Adding anti-memory target-only/slots-only training controls to that Perceiver
  branch also fails the SVAMP32 teacher-forced pre-generation gate.
- Source-generated numeric residue sidecar/router is killed: it avoids control
  leakage but has no clean source-necessary recovery.
- Source final-answer copying and stronger-source source-margin escalation are
  killed for the current frozen SVAMP32 clean IDs.
- Direct source-hidden syndrome readout is killed, including all-layer pooled
  features: all-layer Qwen2.5 features reach only `9/32`, underperform the
  `14/32` target-only floor, preserve only `2/3` target-self rows, and recover
  `0/6` clean source-necessary IDs.
- The first learned query-bottleneck residue predictor over all-layer summary
  tokens is also negative: it matches the all-layer ridge gate at `9/32`,
  preserves only `2/3` target-self rows, and recovers `0/6` clean
  source-necessary IDs.
- The full all-layer source-token learned syndrome probe is negative:
  matched `7/32`, target-only `14/32`, target-self `2/3`, and clean
  source-necessary `0/6`.
- The process-repair / selector stack is negative on the strict SVAMP32
  source-control surface: process repair selected route reaches only `10/32`
  versus `14/32` target-self repair, recovers `1/6` clean residual IDs, loses
  `2/3` target-self repair wins, and selects the target candidate on `32/32`
  examples.
- The target-safe output-aware dynalign selector/repair branch over existing
  SVAMP32 candidates is now killed by an oracle replay. Even an oracle over
  target_self_repair, dynalign salt 1, dynalign salt 2, and query-pool transport
  reaches only `1/6` clean residual IDs, while the matching zero/shuffle control
  oracle also reaches `18/32` and recovers `1/6` clean residual IDs. Another
  selector over these rows is not worth running.
- The cached Qwen2.5-Math-Instruct source variant is also not a rescue surface:
  on frozen SVAMP32 it reaches source `3/32`, target `8/32`, text relay
  `4/32`, source-only over target `2`, and clean source-only after text
  exclusion `2`. This fails the pre-C2C source-surface gate and should not be
  scaled.
- The source-conditioned summary soft-prefix branch is now killed on the
  Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom surface. After calibration to
  source-only matched prefixes, numeric-only distractors, length-normalized
  continuation logprob, and fold-local standardization, it recovers only `1/6`
  clean IDs and has `4/6` clean control leaks.
- The first token-local source cross-attention prefix branch is also negative:
  it recovers `0/6` clean IDs, has `4/6` clean control leaks, and remains
  dominated by label-shuffled, shuffled-source, and target-only controls.
- Surface reselection after these failures ranks `svamp70_live` and
  `svamp70_holdout` highest, but the same cross-attention prefix gate also
  fails on `svamp70_live` with `0/6` matched-only clean IDs and `3/6` clean
  control leaks.
- Fixed source-quality guarded sidecars are also killed by holdout controls:
  the finalish-short-numeric guard reaches `9/70` with clean source-necessary
  `0/2` and clean control union `2/2`.
- Source-trace self-consistency routing is also killed as the next fixed
  sidecar rescue: live CV reaches only `1` clean source-necessary ID with `2`
  accepted harms, and the single holdout clean win survives equation-result
  permutation.

## Main Gaps

1. No deployable positive method.
   The only clean-passing SVAMP object is the C2C-derived syndrome bound.

2. No stable larger-slice positive row.
   GSM seed 0 is source-dependent, but the method is not seed-stable.

3. Exact reproducibility is incomplete from tracked files alone.
   `results/`, `.debug/`, checkpoint tensors, and external repos are ignored;
   many decisive artifacts are local-only and referenced by hashes or commands.

4. Integration testing is thin.
   The unit suite is strong, but `transformers` and vendor integrations are
   mostly stubbed. Real cached-model integration should be a separate marked
   test lane.

5. Cross-family evidence is intentionally blocked.
   The benchmark order correctly says not to widen until the same-family gate is
   cleared.

## Highest-Priority Next Gate

The current live branch is no longer adjacent source-surface scouting, shallow
source-readout tuning, target-safe selector replay, tiny learned prefix
emitters, or another Perceiver/query-memory checkpoint. Those gates have now
failed or become control-explained.

Latest gate update: KVComm/C2C-style cache communication is now the top
baseline branch because it has a strict source-control harness. It is not a
promoted method branch until matched-source performance on a real decision
slice beats zero-source, shuffled-source, and target-only controls or gives a
clear systems tradeoff at comparable accuracy.

The highest-priority next gate is a source-interface reset on the only
remaining strong reusable surface, or a new source/target scout after the stuck
MPS process is cleared. The existing-artifact re-scan ranks:

- `svamp70_live_source`: target `21/70`, source `13/70`, source-only `9`,
  oracle `30/70`
- `svamp70_holdout_source`: target `8/70`, source `8/70`, source-only `6`,
  oracle `14/70`
- adjacent SVAMP70 scouts, GSM70, and SVAMP32 remain below threshold

Do not spend more compute on fixed decoded guards, shallow source-text routers,
tiny prefix emitters, source-token residue readouts, or Perceiver target-memory
checkpoints. The next method branch must be a materially different rate-capped
source interface on `svamp70_live_source` with immediate
`svamp70_holdout_source` validation, or a fresh source/target scout if a
stronger cached source is available. Process repair should remain a target-side
baseline and confound unless a separate source-derived route signal exists.

Current hard blocker: PID `31103` is an orphaned MPS calibration process in
`STAT=UE` and ignores `SIGKILL`. Do not start more MPS jobs until it is cleared.
The source-surface re-scan command that produced the current branch selection
was:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface svamp70_live_source=target_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_live_source \
  --surface svamp70_holdout_source=target_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_holdout_source \
  --surface svamp70_chal171_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal171_240_source \
  --surface svamp70_chal241_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal241_310_source \
  --surface svamp70_chal311_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal311_380_source \
  --surface gsm70_math_source=target_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_gsm70_source \
  --surface svamp32_math_chat_source=target_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,source_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp32_chat_source \
  --min-source-only 6 \
  --output-json results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.json \
  --output-md results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.md
```

Latest cycle update: the Qwen2.5 -> OPT-350m byte-span module-replace proxy is
killed as a decision surface. It had tokenizer mismatch (`shared decoded =
0.9047`, `boundary F1 = 0.9434`) and the harness now supports OPT-style decoder
layers and projected output rows, but the GSM30 surface was too weak:
target-alone `0/30`, source-alone `0/30`, text relay `3/30`, and byte-span
rotalign proxy `0/30` at `525562.6` bytes/example. This is not a decisive kill
of the sequence-aligned sidecar hypothesis; it is a surface failure. The next
attempt must start from a target/text baseline with nonzero headroom before any
source controls are worth running.

Follow-up surface scout: Phi-3 has only weak headroom on GSM30 (`3/30` target,
`1/30` text relay) and SVAMP30 (`5/30` target, `2/30` text relay). TinyLlama is
dead on SVAMP30 (`0/30` target, `0/30` text relay). Do not spend large compute
on cross-family GQA repairs for these exact surfaces unless a stronger baseline
slice is found first.

DeepSeek-R1-Distill-Qwen-1.5B -> Qwen3-0.6B on frozen SVAMP32 is also weak as
an immediate surface. Target-alone reaches `8/32`, while source-alone and text
relay each reach only `5/32`; text adds `2` target-missed IDs, and the
target/text oracle is only `10/32`. C2C is unavailable because no published C2C
artifact is registered for this pair. Do not spend connector or source-control
compute on this pair in the current loop.

Qwen2.5-Math-1.5B -> Qwen3-0.6B is now the strongest same-family decision
surface, but only with chat-template prompting. A no-chat SVAMP16 probe
produced an artificially weak target floor (`0/16`), so it is not
claim-worthy. With chat templates, frozen SVAMP32 reaches target `8/32`, source
`6/32`, text relay `8/32`, and C2C `15/32`. C2C adds `9` target-missed IDs and
the target/C2C oracle reaches `17/32`, versus target/text oracle `11/32`.

The clean C2C-headroom target set is now explicit: source-alone explains `3`
of the `9` C2C-only wins, text relay explains `0`, leaving `6` clean
C2C-headroom targets and `2` target-only-vs-C2C rows to preserve. This is the
current strict-small decision surface. The first deployable probes are
negative: source-only numeric sidecars recover `0/6` clean IDs with only
`26/32` source numeric coverage, and source-hidden ridge probes over last-layer
or all-layer summaries also recover `0/6` clean IDs. Do not widen to larger
slices until a deployable source-derived method clears this exact surface with
full source-destroying controls.

Same-family fallback update: a richer C2C prefill residual projection probe
does not rescue the C2C-mechanism distillation branch. Signed residual
projections reach matched `13/32`, but zero-source, label-shuffle, and
target-only controls reach `14/32`, and clean source-necessary recovery remains
`0/6`. Do not scale C2C summary/projection features without a new token/layer
local objective and anti-cache control.

Latest source-surface update: `Qwen/Qwen2.5-Math-1.5B-Instruct ->
Qwen/Qwen3-0.6B` is weaker than the non-instruct Math source on the frozen
SVAMP32 exact-ID slice. It has only `2` source-only-over-target IDs and should
not receive C2C/sidecar spend. The next gate should not be another adjacent
prompt/source variant; it should be a materially different rate-capped source
interface, such as the smallest real-model sequence-aligned sparse/anchor
sidecar smoke inspired by the quotient/GPA toy results, with zero-source,
shuffle, target-only, and slots-only controls from the start.

Sparse-anchor sidecar update: the first real-model smoke for that branch is
negative. A random sparse anchor projection plus tokenizer-boundary sidecar
reaches `9/32` with `0/6` clean C2C-headroom IDs and one slots-only clean
control hit at an estimated `34` bytes/example. A constrained `14`
bytes/example variant reaches only `7/32`, below the target floor, with `0/6`
clean recoveries. Do not tune this exact projection/top-k implementation
further; the branch only remains live if the feature extractor changes to
fold-local token/span sparse dictionaries or an existing real SAE-adapter lane
is evaluated under the same clean target set and controls.

Target-safe selector update: the dynalign/query-pool selector branch is now
killed on the strict SVAMP32 gate. The target-safe candidate oracle reaches
`18/32` but only `1/6` clean residual IDs, below the required `2/6`, and the
matching source-destroying control oracle also reaches `18/32` with `1/6` clean
residual IDs. The next live branch should be a genuinely learned communication
protocol, starting with a minimal target-conditioned soft-token or learned-query
connector trained against the C2C-over-target_self residual surface with zero,
shuffle, target-only, and slots-only controls from the start.

Qwen2.5-Math learned-connector update: the first target-conditioned
Perceiver/query-innovation checkpoint on the current Qwen2.5-Math -> Qwen3
SVAMP32 C2C-headroom surface is also negative before generation. Calibration
completed with answer-teacher and zero/shuffle plus target/slots anti-memory
controls, but teacher-forced diagnostics at gates `0.125`, `0.15`, and `0.20`
all recover `0/6` matched-only clean residual IDs. Two clean IDs have positive
matched margins, but both are explained by shuffled-source controls. Do not
tune fixed gate, positive weight, answer-teacher weight, or anti-memory weight
on this exact Perceiver memory architecture. The next learned branch needs a
materially different target-query-to-source bottleneck with target-only
learned-prefix and slots-only prefix controls at matched byte/query budgets.

Target-query source-bottleneck update: the first implemented version of that
materially different branch is also negative. The cross-fitted diagnostic uses
target prompt states as queries over source token states and adds zero-source,
shuffled-source, label-shuffle, same-norm-noise, target-only-prefix,
projected-soft-prompt, target-only, and slots-only controls. It reaches only
matched `7/32` versus target-only `8/32`, recovers `0/6` clean residual IDs,
and has no clean source-necessary wins. This kills residue-classifier/readout
variants on the current SVAMP32 C2C-headroom surface. A future learned branch
must train a true source-conditioned soft-prefix or gated cross-attention
objective directly on gold-vs-distractor logprob, not another residue
classifier.

Qwen2.5-Math source-token query-bottleneck update: the non-duplicative all-layer
token bottleneck on the current Math SVAMP32 clean C2C-headroom surface also
fails. It reaches matched `8/32`, exactly the target floor, recovers `0/6`
clean C2C-headroom IDs, and a slots-only control recovers one clean ID. This
kills shallow source-token/source-summary residue prediction on the current
surface unless the feature extractor or objective changes materially.

Fold-local token/span dictionary update: the stricter dictionary version also
fails on the same surface. It has healthy codebook telemetry, with dead atom
rate `0.0000` and mean perplexity `28.5363`, but reaches only `7/32`, below
the `8/32` target floor, and recovers `0/6` clean C2C-headroom IDs. This kills
the current source-readout / sparse-dictionary family on this surface. The next
live branch should move to target-safe output-aware dynalign selector or repair,
not more dictionary/top-k/byte-budget tuning.

Qwen-Math token/layer local follow-up: the new C2C tail-token local residual
query-bottleneck gate also fails. It records per-projector key/value `source`,
`target`, `output`, and `delta` tail tensors, reshaped as `224` tokens of width
`1024`, but matched remains `8/32`, target-only is `8/32`, clean
source-necessary recovery is `0/6`, and slots-only controls recover one clean
ID. This kills C2C summary/projection/tail-local mechanism readouts as a live
branch on this surface unless the supervision objective changes.

Positive strict-small update: switching from C2C-only targets to
source-contrastive targets produces the first live positive source-derived row.
Qwen2.5-Math source-alone has `5` source-only wins over target and `4` clean
source-only wins after excluding text relay. A target/text agreement guard plus
1-byte source residue sidecar reaches `11/32` versus target `8/32`, text relay
`8/32`, and source-alone `6/32`, with `3/4` clean source-necessary recoveries
and `0/4` clean control leakage under zero-source, shuffled-source,
label-shuffle, same-norm noise, target-only, and slots-only controls. This is
not ICLR-ready: it is one 32-example slice, uses text relay as a preservation
guard, lacks paired uncertainty/seed stability, and remains below C2C `15/32`.
Promote to SVAMP70 medium confirmation before any broader claims.

SVAMP70 medium update: the same source-contrastive sidecar stack remains
positive against target/text but is not a headline result. Baselines are target
`21/70`, text relay `22/70`, source-alone `13/70`, and C2C `31/70`. The
guarded sidecar reaches `25/70`, recovers `4/6` clean source-only IDs, and has
`0/6` clean control leakage. Paired deltas are `+0.0571` versus target and
`+0.0429` versus text, but both bootstrap intervals cross zero; versus C2C it
is `-0.0857`. A naive C2C-fallback stack fails (`23/70` with `4/6` clean
control leakage). Do not widen this row to 500 examples until the method either
gets closer to C2C or replaces the text-relay guard with a cheaper preservation
signal that creates a clear systems tradeoff.

Textless guard update: replacing the text-relay agreement guard with a
source/target-only decoded-length guard improves the medium row to `26/70`:
apply the 1-byte source residue sidecar only when the source produces a numeric
prediction and its decoded output is shorter than the target output. This keeps
`4/6` clean source-necessary IDs and `0/6` clean control leakage. Paired delta
versus target is `+0.0714` with bootstrap `[+0.0000, +0.1429]`; versus text it
is `+0.0571` with bootstrap crossing zero; versus C2C it is `-0.0714`. This is
the better systems branch because it removes text relay, but it needs
replication because the length guard is brittle.

Holdout replication update: the fixed length-ratio guard fails on a disjoint
SVAMP70 slice (`chal-101` through `chal-170`). Baselines are source `8/70`,
target `8/70`, text relay `18/70`, and C2C `37/70`. The parameterized
`source_target_len_ratio <= 1.0` sidecar reaches only `10/70`, with clean
source-necessary `0/2` and clean control union `2/2`. This weakens the fixed
hand guard as a live method. Do not scale it directly to 500 examples; the next
live branch must use a learned or cross-validated router, or first discover a
source surface with more clean source-only IDs.

Finalish guard holdout update: the alternative fixed source-quality guard
`finalish_short_numeric` also fails on the same disjoint holdout. Its best
1-byte sidecar row reaches only `9/70`, with clean source-necessary `0/2` and
clean control union `2/2`. This prunes fixed source-quality guarded source
sidecars as the live method family; do not tune thresholds or moduli without a
new router feature family and a frozen holdout gate.

CV router update: a 5-fold decision-stump router over existing source/target
JSONL features can reproduce the original SVAMP70 sidecar row (`25/70`,
`4/6` clean source-necessary, `0/6` clean control union), but it fails the
same disjoint holdout (`6/70`, `0/2` clean source-necessary). This weakens
shallow decoded-feature routers as a robust method and points the next live
branch toward source-surface discovery or stronger source-derived signals.

Source-trace router update: the richer live-CV source-trace router over valid
equations, prompt-number coverage, and source-answer reuse also fails. It has
no standard clean-control leakage, but live CV reaches only `20/70` with `1`
clean source-necessary ID and `2` accepted harms; frozen holdout reaches
`10/70` with `1` clean source-necessary ID, and that ID survives
equation-result permutation. This prunes shallow source-text quality features
as the next rescue for the sidecar branch.

Source-internal diagnostics update: a new sidecar collector can rerun source
generation only and record greedy-generation confidence features, including
chosen-token logprob, entropy, top-1 probability, and top-1/top-2 logit margin.
The two-example MPS smoke passed outside the sandbox with offline caches. This
is the next router feature family to test before another decoded-text guard.

Source-internal confidence router update: live SVAMP70 confidence routing is
clean but too weak (`24/70`, `2` clean source-necessary, `0` clean control
union), and the frozen full-live rule fails the disjoint holdout (`7/70`, `0`
clean source-necessary, `1` accepted harm). This prunes the current confidence
router on the old source-sidecar surface; the next gate is disjoint source
surface discovery, not multi-feature tuning on this slice.

Surface scout update: SVAMP `chal-171` through `chal-240` is not a useful
sidecar decision surface. Source is `8/70`, target is `22/70`, text relay is
`24/70`, source-only over target is only `2`, and clean source-only after text
exclusion is only `1`. Do not spend C2C on this slice for the current branch.

Second surface scout update: SVAMP `chal-241` through `chal-310` has nonzero
clean source mass but still does not clear the predefined surface gate. Source
is `5/70`, target is `10/70`, text relay is `14/70`, source-only over target is
`4`, and clean source-only after text exclusion is `4`. Because raw source-only
is below the `>=6/70` gate and text is much stronger than source, do not spend
C2C here. The cheap sidecar gate now confirms the rejection: the text-relay
agreement guard reaches only `9/70` with clean control leakage, and the
textless shorter-than-target guard reaches only `11/70` with `1/4` clean
source-necessary and `1/4` clean control leakage. The next live gate is GSM70
Math source-surface discovery, while the highest-value method branch after a
surface clears is a rate-capped query/resampler or shared sparse source sidecar
rather than another shallow decoded-feature router.

GSM70 source-surface update: Qwen2.5-Math -> Qwen3 on `data/gsm8k_eval_70.jsonl`
also fails the surface gate. Source is `3/70`, target is `4/70`, text relay is
`6/70`, source-only over target is `3`, and clean source-only after text
exclusion is only `2`. Do not spend C2C or sidecar compute on this slice. The
current live branch should move from more same-pair surface scouting to the
smallest stronger-interface smoke on an existing exact-ID SVAMP surface, or to
a different source/target pair only if a cheap source/target/text scout clears.

Third adjacent SVAMP surface scout update: SVAMP `chal-311` through `chal-380`
also fails the surface gate. Source is `8/70`, target is `21/70`, text relay is
`19/70`, source-only over target is only `3`, and clean source-only after text
exclusion is only `2`. This makes three adjacent same-pair SVAMP70 scouts with
insufficient clean source mass (`chal171-240`, `chal241-310`, and
`chal311-380`). Stop adjacent SVAMP range scouting for Qwen2.5-Math -> Qwen3
unless a new source encoder or prompting hypothesis changes the surface.

Process-repair source-control update: old held-out process-repair rows were
re-audited because they were the strongest historical positive-looking result.
On SVAMP70, matched process repair reaches `38/70`, target self-repair reaches
`35/70`, and matched has `3` wins over target self-repair. The zero-source K/V
control reaches `35/70` and overlaps `1/3` of those matched-only IDs. The
shuffled-source prompt control reaches `37/70` and overlaps `3/3`. The combined
gate therefore has `0` source-specific matched-only IDs after controls. Kill
process-repair selected routes as a source-communication method on this
surface; keep it only as a target-side repair/candidate-diversity baseline. The
next live method branch should be a true source-conditioned soft-prefix or
gated cross-attention logprob objective with matched target-only-prefix,
slots-only, projected-soft-prompt, zero-source, and shuffled-source controls
before generation.

Target-CE prefix-generation update: the proposed true continuation-loss rescue
of that soft-prefix/cross-attention family has now been run and failed. On the
SVAMP32 C2C-headroom surface it gives `0/6` matched-only clean IDs in logprob
and matched generation is weaker than every decoded source-destroying or
target-only control on the six clean IDs. Do not continue tiny prefix-emitter
tuning. The next exact gate is a reusable `latent_bridge` query-innovation
resampler audit for whether true LM CE and generation scoring can be attached
to a larger source-memory interface without a high-risk translator refactor.

Historical source-contrastive promotion rule:

- matched `>=9/32`
- clean source-necessary `>=2/4`
- source numeric coverage `>=26/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, same-norm noise, target-only,
  and slots-only controls have clean union `0/4`

The branch cleared this strict-small source-control surface, but later medium
and disjoint-surface gates showed that the row is unstable. Do not widen it to
500 examples or cross-family benchmarks unless a stronger source surface or
router changes the holdout behavior.

## Engineering Follow-Ups

- Add `pytest -m integration` for tiny cached HF runs covering calibration,
  evaluation, C2C, and KVComm wrappers.
- Replace first-N slices with manifest-backed exact-ID slices including input
  hash, ordered IDs, command, seed, model revision, and output hashes.
- Make the source-control matrix mandatory for every promoted positive row.
- Unify root/package requirements and default local commands around
  `venv_arm64` on this machine.
- Build one tracked artifact manifest for decisive predictions/checkpoints,
  including external repo commits and model revisions.

## Review Inputs

- Local ledger and memos in `paper/`
- Implementation stack in `latent_bridge/`, `scripts/`, and `tests/`
- Experiment artifacts under `results/`, `checkpoints/`, and `.debug/`
- Reference state under `references/`
- Three folder-level subagent audits: paper/story, code/eval, and artifacts

## 2026-04-27 Readiness Update

Current readiness: not ICLR-ready. Estimated distance remains substantial:
the project has headroom surfaces and strong negative controls, but still no
positive communication method that survives disjoint source-destroying controls.

Current live branch: `source_likelihood_sketch` on
`svamp70_live_source` with `svamp70_holdout_source` as frozen validation.

Why this branch is live:

- it is materially different from the killed decoded-feature routers,
  process-repair rows, tiny source-prefix emitters, query-memory rows, and
  Perceiver target-memory rows
- it treats the target candidate pool as decoder side information and sends
  only a rate-capped source-model likelihood preference
- it has a crisp source-control gate: zero-source, shuffled-source,
  label-shuffle, target-only, and slots-only controls must recover zero clean
  source-only IDs

Submission blocker:

- no scientific result yet for this branch because the machine still has an
  orphaned MPS `scripts/calibrate.py` process, PID `31103`, with `STAT=UE`
- do not launch more MPS work until PID `31103` is cleared

Next exact gate:

- run the live and holdout sketch collection plus frozen analyzer commands in
  `paper/svamp70_source_likelihood_sketch_20260427.md`
- promote only if live CV and frozen holdout both clear the predefined pass
  rule; otherwise weaken or kill the branch and move to the next source-surface
  or stronger-interface candidate

## 2026-04-27 Historical Positive Audit Update

The old `rotalign`, `latent_bridge`, and results-folder positives were
re-audited before changing branch priority. The audit is recorded in
`paper/historical_positive_branch_audit_20260427.md`.

Conclusion:

- raw GSM70 dynalign remains a real mechanism clue but is killed as the live
  method because finite repeat seeds do not preserve the seed0 lift
- query-memory and Perceiver target-memory checkpoints stay killed because the
  clean-ID answer-likelihood expansion fails against source-destroying and
  target/slots controls
- process repair remains a target-side baseline/confound, not communication
- the strongest historical direction is the side-information family:
  source-contrastive sidecar plus the C2C-derived syndrome bound

This audit supports keeping `source_likelihood_sketch` as the top branch. It
is the smallest non-duplicative test of whether a source-derived, rate-capped
candidate preference can keep the SVAMP70 live signal while avoiding the
holdout leakage that killed fixed decoded guards.

## 2026-04-27 Collector Hardening Update

The `source_likelihood_sketch` branch remains the live branch. The collector
now supports `--limit` and `--resume` and records command, commit, input
hashes, ordered IDs, ordered-ID hash, and output hash in its markdown readout.

Next gate after PID `31103` is cleared:

1. run the two-example `--limit 2` smoke in
   `paper/svamp70_source_likelihood_sketch_20260427.md`
2. if finite, run the full live and holdout collection commands with `--resume`
3. run the frozen live-to-holdout analyzer

This does not change readiness: still not ICLR-ready until the scientific
live/holdout gate clears.

## 2026-04-27 CPU Smoke Under MPS Blocker

The `source_likelihood_sketch` collector passed a CPU-only two-example smoke
while PID `31103` continued blocking MPS:

- output JSONL:
  `.debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.jsonl`
- output JSONL sha256:
  `863254ecc5110eab3e62efb65ddb31e9472be42513bce6ce1ab44842e1057e9d`
- rows: `2`
- elapsed: `96.06s`
- top labels: `text`, `text`

Readiness impact:

- Tooling risk is lower because the collector can load the source model,
  score continuations, append JSONL rows, and emit provenance/hashes.
- Scientific readiness is unchanged. This is only a micro smoke, and the full
  live/holdout gate still needs MPS after PID `31103` is cleared.

## 2026-04-27 Source Likelihood Sketch Kill

Readiness remains not ICLR-ready.

The `source_likelihood_sketch` live branch is killed on the Qwen2.5-Math ->
Qwen3 SVAMP70 live/holdout surface:

- bare normalized answer mean/sum variants fail live and holdout
- formatted `Answer: {text}` mean logprob has an interesting holdout pass
  (`10/70`, clean source-necessary `2`, control union `0`) but fails live CV
  (`20/70`, clean source-necessary `0`, control union `1`)
- formatted sum-logprob fails live and holdout

The next selected branch is not another likelihood sketch. Post-kill syndrome
bound replays now show that a richer predictor is not justified on this exact
SVAMP70 live/holdout surface: C2C-teacher residues have live headroom but fail
holdout controls, while source-teacher residues recover live clean IDs only by
destroying target-self preservation.

The source-trace router scout also failed and should not be promoted.

Current live branch: none. The next branch is source-surface discovery for a
stronger surface, followed by a bound replay before implementing another
predictor. Stop MPS execution until PID `31103` is cleared; CPU is acceptable
only for tiny smoke/debug work.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

## 2026-04-27 Durable Source-Surface Ranking

Readiness remains not ICLR-ready. Current live method branch: none.

The source-surface selection gate is now durable:

- New ranker: `scripts/rank_source_contrastive_target_sets.py`
- Focused tests: `tests/test_rank_source_contrastive_target_sets.py`
- Output: `results/durable_source_surface_ranking_20260427/source_surface_ranking.json`
- Focused memo: `paper/durable_source_surface_ranking_20260427.md`

The ranker consumes existing `source_contrastive_target_set.json` artifacts and
ranks by clean source-only IDs after controls/baselines, not raw source-only
counts.

Decision:

- `svamp70_live` is the primary next method surface: clean source-only `6/70`,
  raw source-only `9/70`, target/source oracle gain `9/70`.
- `svamp70_holdout` remains the canonical replay gate despite only `2/70`
  clean source-only IDs.
- `svamp70_chal241_310` is only an adjacent falsifier with clean `4/70`.

Recent latent-agent communication references were added in
`references/469_recent_latent_agent_communication_refs.md`. They raise the
baseline bar: the next learned branch should include fixed-budget latent or
activation communication baselines and systems metrics, not only text relay.

Updated next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source MPS scout from
`paper/postkill_historical_cpu_audit_20260427.md`. If it reaches at least six
clean source-only IDs and target/source oracle gain of at least six, run a
zero-init gated latent side-information smoke on `svamp70_live` with
source-destroying controls and activation/latent baselines.

## 2026-04-27 Source-Hidden Query And KVComm Smoke

Readiness remains not ICLR-ready. Current live method branch: none.

CPU evidence weakened direct source-hidden query bottlenecks:

- Command: `scripts/analyze_svamp32_source_latent_syndrome_probe.py` with
  `--probe-model query_bottleneck`, `--query-epochs 2`, `--query-slots 4`,
  `--feature-layers last`, and `--device cpu`.
- Result: `source_latent_syndrome_probe_fails_gate`.
- Matched: `11/32`.
- Zero-source/shuffled-source/label-shuffled/target-only: `14/32`.
- Clean source-necessary IDs: `0`.

CPU tooling smoke for KVComm passed via module invocation:

- Command form: `./venv_arm64/bin/python -m latent_bridge.kvcomm_eval ...`
- One-example CPU smoke wrote `.debug/kvcomm_cpu_smoke_20260427/`.
- Direct script invocation initially failed with `ModuleNotFoundError`; fixed
  `latent_bridge/kvcomm_eval.py` to bootstrap the repo root onto `sys.path`.

Reference update:

- Added `references/470_kv_cache_latent_communication_baselines_refs.md`.
- C2C/KVComm now define the next baseline contract for fixed-budget,
  target-preserving cache communication.

Readiness impact:

- Weakened: direct source-hidden query-bottleneck syndrome readouts.
- Promoted: fixed-budget KV/cache communication baseline as the next executable
  MPS branch after PID `31103` clears.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If clear, run a one-example MPS KVComm smoke or the stronger-source MPS scout,
then scale only if exact ID parity, numeric coverage, and source-destroying
controls are preserved.

If clear, do not run the old `chal311_380` scout recorded in
`paper/svamp70_syndrome_bounds_after_sketch_kill_20260427.md`; those artifacts
already exist and fail the surface gate.

## 2026-04-27 Post-Kill Historical And CPU Audit

Readiness remains not ICLR-ready. Current live branch: none.

The historical positive audit was extended after the `source_likelihood_sketch`
and post-sketch syndrome-bound kills:

- `dynalign_module_replace_residrank16` remains a mechanism clue only; seed
  stability fails and finite repeats do not preserve the seed-0 lift.
- ID-weighted query innovation remains a useful single-ID clue, but still
  recovers only `1/6` clean IDs and does not preserve target-self repair.
- Perceiver/query-memory checkpoints remain killed after six-clean-ID source
  controls.
- Source-contrastive sidecar remains the best historical formulation clue, but
  shallow source-text feature routing does not rescue weak adjacent surfaces.

New CPU-only evidence:

- chal241-310 post-kill source-sidecar CV router fails: best row matches
  `10/70`, clean source-necessary `1`, control clean union `0`, accepted harm
  `1`.
- consolidated existing-surface scan shows `chal311_380` is already available
  and weak: target `21/70`, source `8/70`, source-only `3`, oracle `24/70`.
- existing-artifact CPU mining is exhausted; no remaining CPU-only command can
  promote a positive method.

Focused memo:

- `paper/postkill_historical_cpu_audit_20260427.md`

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, the next MPS command should be a genuinely new
stronger-source scout:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Only run C2C or learned connector work if that scout has ordered ID parity,
high numeric coverage, source-only over target at least `6/70`, and
target-or-source oracle at least target plus `6/70`.

## 2026-04-27 Creative Reference Synthesis

Readiness remains not ICLR-ready. Current live branch: none.

The reference corpus was converted to markdown under
`references/pdf_markdown/` so subagents can inspect the PDF library directly.
The literature sweep changes the next branch priority:

- Candidate-syndrome decoding is promoted as the top CPU-feasible branch:
  source emits a tiny code over target-side candidates, and the target decodes
  against its own side information.
- Zero-init gated query bottlenecks are promoted as the next learned branch
  after MPS clears, because target-self preservation must be built into the
  interface.
- RotAlign/latent-bridge ideas are revived only under anchor-relative sparse
  difference atoms with explicit source-difference zeroing controls.
- Protected-tail quantized residuals are deferred until there is a real
  source-necessary signal worth compressing.

Exact blocker remains a missing positive method plus the local MPS blocker
PID `31103`.

CPU gate result:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir results/candidate_syndrome_decoder_20260427 \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --run-date 2026-04-27
```

Status: `candidate_syndrome_decoder_fails_smoke`.

- Live matched clean source-necessary `1`, target-self harms `17`, control
  clean union `0`.
- Holdout matched clean source-necessary `4`, target-self harms `14`, control
  clean union `0`.

Decision: do not promote the numeric hash-syndrome artifact probe. The next
highest-value branch is zero-init gated query bottlenecks, gated by MPS cleanup
and/or a stronger source surface.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

## 2026-04-27 Byte-Efficient Side-Information Audit

Readiness remains not ICLR-ready. Current live branch: none.

The latest historical/result audit demotes several positive-looking rows to
mechanism clues:

- Sparse K-only and cosine transport showed narrow GSM positives but failed
  later seed stability.
- Raw RotAlign/DynAlign remains killed as a method because seed stability and
  nonfinite checkpoint issues persist.
- Source-contrastive sidecar and candidate-pool syndrome probes remain the best
  low-byte clues, but the old guards/residues are not deployable source
  methods.
- Perceiver/query-memory, shallow source-likelihood, semantic-predicate, and
  numeric hash-syndrome variants are killed on current surfaces.

Updated top branch:

- Learned source-derived syndrome/innovation sidecar decoded against target
  candidate/cache side information.

Required baselines/controls:

- target-alone, source-alone, text/token relay, C2C, KVComm, Q-KVComm-style
  quantized KV where feasible, DroidSpeak-style same-architecture cache reuse
  as a threat model, and target self-repair.
- zero-source, shuffled-source, random same-byte sidecar, target-only,
  slots-only, source-answer-overlap checks, exact-ID parity, numeric coverage,
  paired uncertainty, bytes, latency, generated tokens, and TTFT where
  practical.

New memos:

- `paper/byte_efficient_sideinfo_branch_audit_20260427.md`
- `references/471_byte_efficient_source_sideinfo_refs.md`

Hard blocker:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

PID `31103` still remains present as a `STAT=UE`, `PPID=1` MPS
`scripts/calibrate.py` process even after user-side kill attempts. Do not start
MPS runs until it clears.

Next exact gate after PID clears: run the stronger-source MPS surface scout
recorded in `paper/postkill_historical_cpu_audit_20260427.md`; only build the
learned syndrome/innovation sidecar if that scout has ordered ID parity, high
numeric coverage, source-only over target by at least `6/70`, and
target-or-source oracle at least target plus `6/70`.

## 2026-04-27 Semantic Predicate Sidecar Harness

Readiness remains not ICLR-ready. Current live branch: none.

The semantic-predicate decoder has been hardened into the reusable learned
sidecar gate:

- optional candidate-score sidecar JSONL inputs
- random same-byte sidecar control
- hash-derived non-self shuffled controls
- source-control provenance fields
- tests for target-safe synthetic sidecar recovery

Replay on the existing SVAMP70 live/holdout surfaces remains a method failure:

- live matched: `25/70`, clean source-necessary `3`, accepted harm `0`
- live random same-byte sidecar: `17/70`, clean source-necessary `1`,
  accepted harm `7`
- holdout matched: `9/70`, clean source-necessary `0`, accepted harm `0`

Decision: old semantic-predicate branch is killed more decisively. Keep the
hardened harness only for a genuinely new frozen learned sidecar after a
stronger source surface clears.

New memo:

- `paper/semantic_predicate_sidecar_harness_20260427.md`

Next exact gate remains the MPS-blocker check, then the stronger-source scout
from `paper/postkill_historical_cpu_audit_20260427.md`.

## 2026-04-27 MPS Micro Stronger-Source Surface Gate

Readiness remains not ICLR-ready. The hard MPS blocker has cleared:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Result: `blocked=false`, PID `31103` absent.

The first MPS micro scouts were operationally clean but scientifically negative:

- `Qwen/Qwen2.5-7B-Instruct -> Qwen/Qwen3-0.6B`, SVAMP8:
  source `2/8`, target `2/8`, text relay `1/8`, clean source-only `1`,
  answer-unexplained clean in pool `0`.
- `Qwen/Qwen2.5-Math-1.5B -> Qwen/Qwen3-0.6B`, SVAMP8:
  source `2/8`, target `2/8`, text relay `2/8`, clean source-only `0`,
  answer-unexplained clean in pool `0`.

Decision: MPS generation is usable again, but these micro rows do not support a
positive communication claim. The 7B row has one clean source-only ID after text
relay, but it is explained by the source final answer.

Current live branch: stronger-source answer-masked surface discovery on a
different slice/model, not a learned sidecar yet.

Next exact gate: run a bounded 7B or Math-7B discovery slice chosen for
source/target disagreement and reject unless `answer_unexplained_clean_in_pool`
is nonzero under exact ID parity, text relay, and source-destroying controls.

## 2026-04-27 Disagreement Surface And JEPA Anti-Collapse

Readiness remains not ICLR-ready. The selected 7B disagreement surfaces improved
diagnostic sharpness but did not unlock a positive branch:

- clean6 selected from historical SVAMP70 clean source-only IDs:
  target `0/6`, source `1/6`, text relay `1/6`, answer-unexplained clean in pool
  `0`.
- disagreement12 union from live/chal/holdout historical clean IDs:
  target `0/12`, source `4/12`, text relay `4/12`, clean source-only `2`,
  answer-unexplained clean in pool `0`.

Decision: do not promote a learned sidecar or latent connector. The useful
source rows are still explained by final numeric answers or are outside the
target-side candidate pool.

JEPA/LeJEPA/V-JEPA literature was added as design guidance, not evidence:
future connector training should use answer-masked dual source views, frozen
target latent targets, matched-source margin over zero/shuffled/target-only/
slots-only controls, target-preservation loss, and collapse telemetry
(`std_min`, effective rank, covariance off-diagonal, Barlow-style
cross-correlation).

Current live branch: CPU-only answer-masked source/answer-only diagnostics and
collapse telemetry. Next exact gate is an answer-likelihood smoke on live and
holdout source surfaces before any more MPS generation.

## 2026-04-27 No-Source Candidate Surface Gate

Readiness remains not ICLR-ready. Current live branch: target-only sampled
candidate surface plus source-derived selector.

The existing no-source process-repair/zero-source-KV surface is now audited as a
candidate pool rather than a communication method:

- zero-source candidate surface target-side oracle: `48/70`
- target baseline: `21/70`
- source baseline: `13/70`
- clean source-only IDs after no-source baselines: `3`
- clean source-only IDs whose gold answer is in the no-source pool: `0/3`

Decision: kill source-derived selection over the existing zero-source candidate
surface. There is no clean source-necessary target for a selector to recover.

A CPU target-only sampling smoke on those remaining three IDs recovered one
ID:

- recovered clean ID: `14bfbfc94f2c2e7b`
- target-only sampled candidate oracle: `1/3`
- artifact: `results/target_only_sampling_clean3_20260427/target_only_samples.md`

This is not positive source-communication evidence. It only revives the next
bounded branch: sampled target-only candidate generation followed by a
source-derived selector with zero/shuffled/random/target-only/slots-only
controls.

New memo:

- `paper/no_source_candidate_surface_and_target_sampling_20260427.md`

Next exact gate: materialize the sampled clean3 target-set rows and run the
source candidate sidecar controls on that sampled pool. Promote only if matched
source selects `14bfbfc94f2c2e7b` and all source-destroying controls miss it.

## 2026-04-27 Target-Pool Sidecar Hardening

Readiness remains not ICLR-ready. Current live branch: none.

The sidecar harness now prevents a major control leak: target-side candidate
pools exclude source-only values by default. Candidate-score sidecars may only
map explicit values that already appear in the target-side pool, unless a
future method explicitly accounts for transmitting the value as payload bytes.

Additional hardening:

- random same-byte controls preserve declared learned-sidecar bit budgets
- sidecar-shaped `target_only_sidecar` and `slots_only_sidecar` controls are
  now first-class conditions
- sidecar JSONL rejects duplicate IDs
- supplied sidecars must exactly cover target-set reference IDs
- summaries report accepted help and accepted clean-source help

Replay on existing SVAMP70 live/holdout artifacts is a stronger kill:

- live matched: `24/70`, clean `3`, accepted harm `1`
- live random same-byte sidecar: `16/70`, clean `0`, accepted harm `9`
- live target-only sidecar: `21/70`, clean `0`, accepted harm `0`
- live slots-only sidecar: `21/70`, clean `0`, accepted harm `0`
- holdout matched: `9/70`, clean `0`
- control clean union: `0`

New memo:

- `paper/semantic_sidecar_target_pool_hardening_20260427.md`

Next exact gate remains the MPS-blocker check, then the stronger-source scout
from `paper/postkill_historical_cpu_audit_20260427.md`.

## 2026-04-27 SVAMP32 Full32 Target Sampling Reachability

Readiness remains not ICLR-ready. The current story is now sharper: broad
target/no-source sampling creates a large receiver-side candidate pool, but it
does not by itself create new clean C2C residual communication surface.

New evidence:

- `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.md`
  reports raw target/no-source candidate oracle `14/32` with full numeric
  coverage (`256/256`).
- `results/svamp32_target_sampling_full32_s8_20260427/reachability.md` reports
  raw sample oracle gain `7`, C2C clean residual in pool `2/6`, C2C
  teacher-only in pool `4/9`, and duplicate nonempty row fraction `0.582`.
- `results/svamp32_target_sampling_full32_s8_20260427/headroom.md` reports the
  merged target-side oracle with text relay plus samples as `18/32`, but the
  remaining clean source-only IDs have gold in target-side pool `0/2`.

Decision:

- target/no-source candidate-pool discovery passes as a receiver-headroom
  diagnostic.
- the selector surface is not expanded, because the only C2C-clean residual IDs
  reached are still `3e8a5691f5443495` and `575d7e83d84c1e67`, already reached
  by the clean6 `s16` gate.
- do not spend another cycle on deterministic numeric/process selectors for
  this pool.

Current live branch:

- bounded learned source-conditioned candidate generator or frozen-latent,
  rate-capped connector with JEPA/LeJEPA/V-JEPA-style anti-collapse telemetry.

Next exact gate:

- run the smallest learned connector/generator smoke that can use the full32
  no-source pool as a target-prior baseline. Promote only if matched source
  recovers at least `1` C2C-clean source-necessary ID while zero-source,
  shuffled-source, target-only/slots-only, and random same-byte controls recover
  `0`, with no target-correct harm and byte/latency accounting.

Update `2026-04-27`: the literature-to-method sprint promotes the
source-private evidence-packet / candidate-syndrome benchmark as the highest
probability one-month path. JEPA/V-JEPA/LeJEPA and VLM connector work remain
important for anti-collapse and second-stage learned adapters, but they should
not lead until the private benchmark exposes residual IDs. A deterministic
source-private evidence packet smoke now passes the contract:
`matched_syndrome=128/128`, target/no-source `32/128`, zero/shuffled/random/
answer-only/answer-masked/target-only-sidecar controls `32/128`, matched-byte
structured text `32/128`, and full structured text oracle `128/128` at `13`
bytes. This is harness evidence only, not a paper claim. The live branch is now
`source_private_evidence_packet_strict_small_20260428`: instantiate the same
contract with LLM-generated target candidates and source packets on `100-200`
frozen private-evidence examples, with byte/latency accounting and full
source-destroying controls.

Update `2026-04-28`: `source_private_evidence_packet_strict_small_20260428`
passes as a strict-small deterministic source-private protocol gate. On `160`
frozen private-evidence QA examples, a matched binary syndrome reaches
`160/160` at every budget `2/4/8/16/32` bytes, while target/no-source,
matched-byte structured text, zero-source, shuffled-source, random same-byte,
answer-only, answer-masked, target-derived, and wrong-salt same-source controls
stay at `40/160`. Full structured text and full private evidence oracles are
`160/160`. Exact ID parity holds with SHA256
`3a65952ba323a8896906863f1be4e83400a6cea00ab1f18bbf58cb8e7611b19c`. This
promotes the source-private candidate-syndrome branch from smoke to strict-small
protocol pass, but the paper is still not ICLR-ready: packet production and
decoding are still protocol-shaped rather than model-mediated. The next exact
gate is `source_private_evidence_packet_llm_packet_20260428`, using the same
frozen IDs and controls while replacing deterministic source packet production
with model-produced packets.

Update `2026-04-28`: the source-private hidden-repair packet smoke now replaces
synthetic trace signatures with actual hidden execution evidence. On `64`
frozen Python repair examples, deterministic 2-byte repair packets reach
`64/64`, while target/no-source and zero/shuffled/random/answer-only/
answer-masked/target-derived/matched-byte-text controls stay at `16/64`. A
Qwen3-0.6B source model also emits valid helper-line packets for `64/64`
examples, with the same `64/64` matched versus `16/64` controls result. This
promotes the live branch to model-mediated hidden-repair smoke, but it remains
protocol-assisted: helper-line diagnostics and candidate metadata are still
doing real work. The next exact gate is
`source_private_hidden_repair_packet_cross_model_20260428`, followed by a
no-helper or weakened-helper hidden-repair gate before any ICLR-level claim.

Update `2026-04-28`: the hidden-repair packet branch now has cross-model smoke
evidence. On the same frozen `64` examples, Qwen2.5-0.5B reaches `63/64`,
Qwen3-0.6B reaches `64/64`, and Phi-3-mini reaches `64/64`, while all
target-only and source-destroying controls remain at `16/64`. TinyLlama fails
with `0` valid packets and stays at target-only, which is useful as a negative
capability row. This removes the one-model prompt-artifact objection, but the
paper is still not ready because helper-line diagnostics and candidate metadata
remain part of the protocol. The next exact gate is
`source_private_hidden_repair_packet_weakened_helper_20260428`.

Update `2026-04-28`: weakened-helper evidence improves the claim boundary.
Removing the copied helper line leaves Qwen3 at `63/64`; removing both the
copied helper line and the hint while keeping the private `REPAIR_DIAG` trace
leaves Qwen3 at `50/64` and Phi-3-mini at `64/64`, with controls still at
`16/64`. Removing the trace itself drops Qwen3 to target-only (`16/64`) with
`0` valid packets. The live story is now explicit source-private tool-trace
handoff, not raw-log repair inference. Next gate:
`source_private_hidden_repair_packet_strict_small_20260429`.

Update `2026-04-29`: `source_private_hidden_repair_packet_strict_small_20260429`
passes. On `160` frozen hidden-repair examples, deterministic packets reach
`160/160` at 2 bytes while target-only remains `40/160` and controls are
`40-41/160`. With model-produced `trace_no_hint` packets, Qwen3 reaches
`127/160` and Phi-3-mini reaches `160/160`; target-only is `40/160` and best
source-destroying control is `41/160`. Qwen3 `raw_log_no_trace` drops to
`40/160` with `0` valid packets. The repo now has a strict-small positive
method candidate: explicit source-private tool-trace packet handoff. It is
still not ICLR-ready until medium/larger confirmation with paired uncertainty.

Update `2026-04-29`: `source_private_hidden_repair_packet_medium_20260429`
passes on `500` frozen examples with paired bootstrap intervals. Qwen3
`trace_no_hint` reaches `404/500`, Phi-3-mini reaches `500/500`, target-only is
`125/500`, and best source-destroying control is `126/500`. Bootstrap 95%
paired deltas over target-only are `[0.516, 0.600]` for Qwen3 and
`[0.714, 0.788]` for Phi-3. Qwen3 `raw_log_no_trace` returns to target-only
with `0` valid packets. The live branch is now medium-confirmed as explicit
source-private tool-trace packet handoff. Remaining blocker: held-out repair
families and seed repeats.

Update `2026-04-29`: held-out repair-family confirmation passes. A disjoint
eight-family held-out benchmark with `500` examples gives Qwen3
`trace_no_hint` `461/500`, Phi-3-mini `500/500`, target-only `125/500`, and
best control `129/500`. Paired bootstrap 95% deltas over target-only are
`[0.632, 0.712]` for Qwen3 and `[0.710, 0.788]` for Phi-3. Qwen3
`raw_log_no_trace` returns to `125/500` with `0` valid packets. The method is
now medium-scale plus held-out-family positive. It remains not ICLR-ready until
seed repeats and baseline/framing are complete.

Update `2026-04-29`: seed repeats pass across four frozen `500`-example
surfaces: core seeds `29` and `31`, held-out seeds `30` and `32`. All `8/8`
primary `trace_no_hint` rows pass for Qwen3 and Phi-3; all `4/4`
`raw_log_no_trace` rows fail as intended with matched accuracy at target-only
and `0` valid packets. The minimum paired-bootstrap lower bound over target-only
is `0.516`, and the minimum lower bound over best source-destroying control is
`0.506`. The repo now has a seed-stable positive method candidate. Remaining
ICLR blocker: reviewer-facing baseline/system package and precise claim
framing.

Update `2026-04-29`: `source_private_tool_trace_baseline_pack_20260429`
consolidates the live evidence into a reviewer-facing package. Covered controls
include target-only/wrapper, zero-source, shuffled-source, random same-byte,
answer-only, answer-masked, target-derived sidecar, matched-byte truncated
hidden-log text, full hidden-log relay, full diagnostic oracle, trace removal,
held-out families, and seed repeats. The exact claim should now be scoped as:
explicit source-private tool-trace packets communicate hidden execution
evidence to a target-side candidate decoder. Remaining reviewer-risk rows:
matched-byte JSON/free-text relay, helper-only no-log oracle, trace-component
masking, candidate/selector separation, and a second target-family row.

Update `2026-04-29`: `source_private_tool_trace_paper_claim_draft_20260429`
sets the paper boundary. The live paper story is now: source-private
tool-trace packets are a compact communication interface from a private source
agent to a target-side candidate decoder. Across four frozen `500`-example
surfaces, Qwen3 reaches `0.808-0.924`, Phi-3 reaches `1.000`, target-only is
`0.250`, best source-destroying controls stay `0.252-0.258`, and trace removal
returns Qwen3 to target-only with `0` valid packets. This is close to an ICLR
positive-method claim if framed narrowly. The next exact gate is
`source_private_tool_trace_reviewer_risk_rows_20260429`, focused on
matched-byte structured JSON/free-text relays, helper-only/no-log oracle,
trace-component masking, and explicit candidate-pool versus selector reporting.

Update `2026-04-29`: `source_private_tool_trace_reviewer_risk_rows_20260429`
passes on representative `500`-example core and held-out surfaces. At the
`2`-byte paper packet budget, matched packets are `1.000`, target-only is
`0.250`, best source-destroying control is `0.254`, best reviewer negative
control is `0.250`, and min reviewer oracle is `1.000` on both surfaces.
Matched-byte JSON/free-text relays, helper-template/no-log, and diagnostic-
masked full-log rows stay at target-only; expected/actual-masked and
test-name-masked full logs remain oracles. At `32` bytes, structured JSON/free
text become oracles, so the paper should report this as a rate tradeoff. The
next exact gate is `source_private_tool_trace_final_table_20260429`.

Update `2026-04-29`: `source_private_tool_trace_final_table_20260429`
integrates model-mediated rows, deterministic controls, reviewer-risk rows,
systems bytes/tokens, and candidate-pool versus selector separation into one
paper-facing evidence table. The repo now has evidence-ready support for the
scoped positive claim: compact explicit source-private tool-trace packets
communicate hidden execution evidence to a target-side candidate decoder. The
next exact gate is `source_private_tool_trace_paper_skeleton_20260429`. The
main remaining reviewer-strategy question is whether to add an optional learned
target-decoder row or explicitly scope it as future work.

Update `2026-04-29`: `source_private_tool_trace_paper_skeleton_20260429`
drafts the paper structure around the scoped positive claim. The proposed
paper should be framed as rate-capped source-private communication with
interpretable tool-trace packets, not learned latent transfer. The next exact
gate is `source_private_tool_trace_skeleton_review_20260429`: skeptical review
of overclaiming, novelty, baselines, leakage, and whether an optional
LLM-mediated target-decoder row is worth adding before full paper drafting.

Update `2026-04-29`: `source_private_tool_trace_skeleton_review_20260429`
identifies the remaining submission risk as novelty/protocol shape: reviewers
may view the deterministic decoder as coded-label lookup. The skeleton was
patched to use source-private evidence-communication wording and to make the
next gate `source_private_tool_trace_target_decoder_smoke_20260429`: one cheap
LLM-mediated or learned target-side selector row on a frozen slice, with the
same source-destroying controls. If it fails, learned target decoders should be
listed as a limitation rather than blocking the scoped protocol claim.

Update `2026-04-29`: `source_private_tool_trace_target_decoder_smoke_20260429`
passes as a small ablation. Qwen3-0.6B acting as the target-side selector
reaches `0.688` matched versus `0.250` target/control on a 16-example core
slice, and `0.750` matched versus `0.250` target and `0.281` best control on a
32-example held-out slice. This does not replace the large deterministic
evidence, but it reduces the novelty risk that the method is only hand-coded
lookup. The next exact gate is
`source_private_tool_trace_paper_sections_20260429`.

Update `2026-04-29`: `source_private_tool_trace_paper_sections_20260429`
drafts section-level manuscript text in memo form: abstract, introduction,
problem setup, method, benchmark, results, controls, systems, target-decoder
smoke, interpretability, limitations, and conclusion. The repo is now ready to
move from research memos into paper-source drafting. The next exact gate is
`source_private_tool_trace_paper_draft_20260430`.

Update `2026-04-30`: `source_private_tool_trace_paper_draft_20260430` creates
a full markdown paper draft with figure/table placeholders and explicit claim
boundaries. The manuscript is now draft-present but not submission-ready:
remaining work is LaTeX/source conversion or figure/table asset generation,
plus final skeptical review for overclaiming, related-work framing, and
baseline presentation. The next exact gate is
`source_private_tool_trace_latex_or_figures_20260430`.

Update `2026-04-30`: `source_private_tool_trace_draft_review_20260430`
completed a skeptical review of the full markdown draft. The draft was patched
to soften broad claims and emphasize frozen synthetic surfaces, the benchmark's
hidden diagnostic evidence, and direct auditability rather than general
latent-transfer claims. Submission blockers remain: setup/rate-curve figures,
count-augmented tables, concrete citations, and a choice between scaling the
target-decoder smoke or listing it as an ablation/limitation.

Update `2026-04-30`: `source_private_tool_trace_latex_or_figures_20260430`
generates the setup diagram and rate-curve assets. The rate curve is now an
actual SVG/CSV artifact and should be in the main paper because it shows the
honest systems tradeoff: compact packets work at `2` bytes, while structured
JSON/free-text relays become oracles at `32` bytes. The next exact gate is
`source_private_tool_trace_latex_20260430`.

Update `2026-04-30`: `source_private_tool_trace_latex_20260430` creates a
project-specific ICLR LaTeX draft and bibliography under `paper/iclr2026/`.
The source has not yet been compiled. The next exact gate is
`source_private_tool_trace_latex_compile_20260430`, including any needed
SVG-to-PDF conversion and bibliography cleanup.

Update `2026-04-30`: `source_private_tool_trace_latex_compile_20260430`
passes. The live branch is now an explicit source-private tool-trace packet
handoff with large frozen-slice evidence and a compiled ICLR-style paper
source. The paper story is scoped: compact source-private packets communicate
hidden diagnostic evidence to a target-side candidate decoder at `2`
bytes/tokens, while zero/shuffled/random/answer-only/answer-masked/
target-derived controls stay near target-only and structured text relays
become competitive only at larger budgets. The LaTeX source compiles to
`paper/iclr2026/source_private_tool_trace.pdf` (`7` pages, `226624` bytes) with
no overfull boxes, undefined references, or citation warnings. Remaining
blockers are final skeptical review, target-decoder scale-up decision, and
final table/figure/citation polish.

Update `2026-04-30`: `source_private_tool_trace_final_review_20260430` applies
the final skeptical-review framing patch. The manuscript now explicitly frames
the result as an explicit diagnostic-code protocol benchmark, not learned
latent transfer or unconstrained program repair; the target-decoder row is
scoped as a model-mediated protocol-decoder smoke ablation; related work now
covers decoder-side information, multi-agent handoff, prompt compression,
C2C/KVComm-style latent/cache communication, and test-guided repair framing.
The next exact gate is `source_private_tool_trace_submission_polish_20260430`:
line-level PDF/source polish, citation metadata verification, optional appendix
example, and final scoped-submit decision.

Update `2026-04-28`: submission-polish follow-up verified primary-source
metadata for the new citations and corrected Repair-R1 to arXiv `2507.22853`.
The manuscript now uses more precise line-level language for near-target-only
controls, constructional target prior, smoke-ablation target decoding,
matched--target confidence intervals, candidate-selection scope, and artifact
provenance. A representative appendix example now illustrates a target-prior
failure corrected by the matched private packet. Remaining blocker: final
scoped-submit decision.

Update `2026-04-28`: the final scoped-submit decision is now documented in
`paper/source_private_tool_trace_submission_decision_20260428.md`. Proceed with
the scoped protocol-method submission path; do not run the optional target-
decoder `n=160` scale-up unless the main claim is expanded to an LLM target
receiver. The live story is explicit source-private diagnostic-code packets for
candidate selection with decoder side information, not learned latent transfer
or raw-log repair. The remaining blocker is a final human PDF/source read and
conference-form check.

Update `2026-04-28`: the final human PDF/source hygiene read is complete. The
paper source now uses local figure paths, public manifests avoid absolute user
paths, and target-decoder smoke wording is limited to a sanity check. The source
upload bundle
`paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`
was built and compile-tested from a scratch extraction. The scoped paper package
is ready for manuscript upload. Remaining non-paper blocker: for an external
artifact release, archive or force-add decisive raw JSON/JSONL inputs/
predictions that are currently summarized by tracked manifests but ignored
under `results/` or `.debug/`.

Update `2026-04-28`: the external artifact-release archive is now built at
`paper/artifacts/source_private_tool_trace_artifacts_20260428.zip` (`7548312`
bytes, SHA256
`64153e44dd5b41a30e54ffa5cdb0d95ca5498c2345c13da015a9f9f076c0121f`). It
contains the compiled PDF, compile-tested source bundle, decisive raw
JSON/JSONL results, target-decoder input surfaces, figure/rate data, and
readout memos. Archive integrity and local-path hygiene checks passed. The repo
is now ready for manuscript upload plus artifact handoff; remaining distance is
external venue/upload mechanics, not additional method evidence.

Update `2026-04-28`: final external-handoff instructions are documented in
`paper/source_private_tool_trace_external_handoff_20260428.md`, including upload
paths, hashes, claim-boundary language, and post-upload sanity checks. The
readiness status is upload-ready. The next gate is only to record the actual
submission/artifact upload confirmation and any portal-required changes.

Update `2026-04-28`: local work is complete for submission handoff. External
submission confirmation is blocked on portal/artifact-host state that is not
available from the repo. The blocker and required confirmation fields are
recorded in `paper/source_private_tool_trace_submission_confirmation_20260428.md`.

Update `2026-04-28`: added
`paper/source_private_tool_trace_upload_checksums_20260428.sha256` for the
three external upload files and linked it from the handoff/confirmation memos.
The repo remains upload-ready and externally unconfirmed.

Update `2026-04-28`: artifact-host readiness is documented. Public GitHub
release hosting is available from the authenticated CLI, but it was not
published because the current repo is public and non-anonymized. The exact
public-release command and anonymous-host alternative are recorded in
`paper/source_private_tool_trace_artifact_host_options_20260428.md`.

Update `2026-04-28`: the double-blind artifact route is now preferred and ready.
The anonymous artifact archive is
`paper/artifacts/source_private_tool_trace_artifacts_anonymous_20260428.zip`
(`7553609` bytes, SHA256
`02b1dbd73dea1332976e60a255def4a470f6c91416f2603cbd8631270be3790a`) with
checksum sidecar
`paper/source_private_tool_trace_anonymous_upload_checksums_20260428.sha256`.
Integrity and anonymity string audits passed.

Update `2026-04-28`: added the anonymous transfer convenience bundle
`paper/artifacts/source_private_tool_trace_anonymous_submission_bundle_20260428.zip`
(`8137764` bytes, SHA256
`6d771cfc41ad4acfb8d500a6841e06e28781b3ce2549fc368dff0a1ed666e377`). The
bundle passed archive integrity, internal checksum verification, and
identity/local-path string audit. It is for moving the upload files together;
the venue may still require separate PDF/source/artifact uploads.

Update `2026-04-28`: final local upload audit passed. Checksums, archive
integrity, extracted-tree identity scan, and binary/string identity scan over
the exact upload files all passed. The repo is locally complete for anonymous
submission; remaining work is external confirmation only.

Update `2026-04-28`: added `final/` as the consolidated reproducibility and
finalization folder. It includes the upload payload, paper source/PDF,
source-private code and tests, relevant result directories, references, and
checksums. The repo remains locally complete for anonymous submission; `final/`
is a staging copy for handoff, not a change to the scientific claim.

Update `2026-04-28`: latest-model/MoE generalization remains unproven. The new
matrix in `results/source_private_latest_model_matrix_20260428/` selects
Qwen3.5 `0.8B/2B/4B` as local small rows and Qwen3.5/Qwen3.6 `35B-A3B` plus
FP8 as off-machine MoE falsification rows. Local Qwen3.5 execution is blocked by the
repo's current Transformers version lacking `qwen3_5` support. This does not
change readiness of the current scoped submission, but it defines the next
evidence gate if we want a stronger model-generalization claim.

Update `2026-04-28`: the Qwen3.5 local blocker is partially cleared. After
upgrading the repo-local stack to `transformers==5.7.0`,
`tokenizers==0.22.2`, and `huggingface_hub==1.12.0`, `Qwen/Qwen3.5-0.8B`
passes the source-private hidden-repair packet gate on CPU at n16 and n64:
matched packets `1.000`, target/control floor `0.250`,
matched-minus-best-control `+0.750`, packet valid rate `1.000`, and exact-ID
parity true. Apple MPS still fails before generation with a hybrid-attention
matmul shape error, so the evidence is CPU-only. Readiness for the scoped
submission stays upload-ready; readiness for a latest-model/MoE claim improves
from planned to latest-small smoke/confirmation only. Next exact gate:
Qwen3.5-0.8B n160 with a seed repeat and one non-Qwen n16 row, followed by
off-machine Qwen3.6 MoE/FP8 only if the small rows remain clean.

Update `2026-04-28`: latest-small readiness improves again. Qwen3.5-0.8B now
passes CPU trace-no-hint n160 seed29 with matched `1.000`, target-only `0.250`,
best source-destroying control `0.256`, packet valid rate `1.000`, and exact-ID
parity true. The first non-Qwen positive also exists: Granite-3.3-2B-Instruct
CPU copied-helper n64 reaches `51/64 = 0.797` versus target/control `16/64 =
0.250`, packet valid rate `0.734`, and exact-ID parity true. OLMo-2-0425-1B is
negative with `0` valid packets, and Granite MPS is backend-blocked. Readiness
for the scoped submission is unchanged; readiness for a stronger
model-generalization section is now latest-small medium plus non-Qwen
confirmation, still short of seed repeat, Granite n160, and MoE/FP8 rows.

Update `2026-04-28`: latest-small evidence is now seed-stable at medium scale.
Qwen3.5-0.8B CPU trace-no-hint n160 passes on seeds 29 and 31 with matched
`160/160`, target/control floor near `0.250`, packet valid rate `1.000`, and
exact-ID parity true. Granite-3.3-2B copied-helper CPU n160 passes at
`128/160 = 0.800` versus target-only `0.250` and best control `0.256`, while
Granite trace-no-hint n64 is positive but weaker (`37/64 = 0.578`). OLMo remains
a negative behavioral row. The scoped submission remains upload-ready; the
latest-model/cross-family section can now safely discuss seed-stable Qwen3.5 and
medium Granite prompt-contract sensitivity, but still cannot claim MoE or
prompt-invariant cross-family generalization.

Update `2026-04-28`: Qwen3.5 latest-small breadth improves from one size to two.
Qwen3.5-2B CPU trace-no-hint n16 passes with matched `16/16`, target/control
`4/16`, packet valid rate `1.000`, and exact-ID parity true. This is only a
smoke row, but it supports the same-generation scaling story alongside
Qwen3.5-0.8B n160 seed stability. Added a Qwen3.6 MoE/FP8 runbook with exact
CUDA and endpoint-wrapper requirements; MoE readiness remains planned until
those off-machine n32 rows exist.

Update `2026-04-28`: Qwen3.5-2B now has n64 confirmation, not just smoke.
CPU trace-no-hint n64 reaches matched `64/64`, target/control `16/64`, packet
valid rate `1.000`, and exact-ID parity true. This strengthens the latest-small
same-generation breadth claim, while still leaving Qwen3.5-4B and Qwen3.6 MoE
unrun.

Update `2026-04-28`: Qwen3.5-2B now has n160 confirmation. CPU trace-no-hint
n160 seed29 reaches matched `160/160`, target-only `40/160`, best
source-destroying control `41/160`, packet valid rate `1.000`, exact-ID parity
true, matched-minus-best-control `+0.744`, and p50 CPU latency `13878 ms`.
This strengthens the latest-small same-generation section from one Qwen3.5 size
to two n160-confirmed sizes. The scoped submission remains upload-ready; the
latest-model addendum is stronger but still not MoE-claim-safe until
Qwen3.6-35B-A3B/FP8 rows run off-machine under the same controls.

Update `2026-04-28`: added `source_private_codebook_remap_gate_20260428`.
The deterministic all-family `500`-example gate passes across three remapped
diagnostic codebooks (`seeds 29/31/37`) and low-rate budgets `2/4/8/16`.
Exact IDs and public candidate labels are identical across seeds; codebook
hashes differ; matched packets remain `1.000`; no-source remains `0.250`; best
source-destroying controls stay at `0.250-0.256`; same-budget JSON/free-text
and diagnostic-masked controls remain `0.250`. This closes one reviewer-risk
gap around fixed-codebook memorization, but does not change the need for MoE,
FP8, and stronger non-Qwen source-emitter rows.

Update `2026-04-28`: Qwen3.5-4B now has a local CPU smoke row. CPU
trace-no-hint n16 seed29 reaches matched `16/16`, target-only/best control
`4/16`, packet valid rate `1.000`, exact-ID parity true,
matched-minus-best-control `+0.750`, and p50 CPU latency `32485 ms`. This
strengthens the latest-small same-generation section from two to three Qwen3.5
sizes, but the 4B row is not yet medium confirmation and does not reduce the
need for Qwen3.6-35B-A3B/FP8 MoE rows.

Update `2026-04-28`: Qwen3.5-4B now has local CPU n64 confirmation. CPU
trace-no-hint n64 seed29 reaches matched `64/64`, target-only/best control
`16/64`, packet valid rate `1.000`, exact-ID parity true,
matched-minus-best-control `+0.750`, and p50 CPU latency `27188 ms`. This
upgrades the latest-small same-generation section to three Qwen3.5 sizes with
at least medium evidence for 2B/4B and seed-stable n160 evidence for 0.8B. It
still does not reduce the need for Qwen3.6-35B-A3B/FP8 MoE rows.

Update `2026-04-28`: cross-family/latest-model evidence strengthens. After
downloading `google/gemma-4-E2B-it`, CPU trace-no-hint n64 seed29 reaches
matched `64/64`, target-only/best control `16/64`, packet valid rate `1.000`,
exact-ID parity true, matched-minus-best-control `+0.750`, and p50 CPU latency
`2179 ms`. Granite-3.3-2B-Instruct also now has a strict trace-no-hint CPU n160
row: matched `101/160 = 0.631`, target-only `40/160`, best control `41/160`,
packet valid rate `0.537`, exact-ID parity true, and p50 CPU latency `2816 ms`.
This weakens the prompt-engineered non-Qwen objection, but the full-paper
generalization blocker remains Qwen3.6-35B-A3B/FP8 MoE under the same controls.

Update `2026-04-28`: the Qwen3.6 MoE blocker is now execution-access rather
than missing harness support. Added an OpenAI/vLLM-compatible endpoint runner
for the hidden-repair packet gate, with tests verifying that endpoint packets
flow through the same evaluator and source-destroying controls. The next exact
gate is `Qwen/Qwen3.6-35B-A3B` endpoint n32 seed29, followed by
`Qwen/Qwen3.6-35B-A3B-FP8` endpoint n32 if the dense-MoE row passes.

Update `2026-04-28`: remote execution is disallowed, so the next best local
architecture-diversity hardening gate widened Gemma 4 E2B. `google/gemma-4-E2B-it`
MPS trace-no-hint n160 passes on seeds 29 and 31 with matched `160/160`,
target-only `40/160`, source-destroying controls at `40/160` to `41/160`,
packet valid rate `1.000`, exact-ID parity true, and p50 MPS packet latency
`821 ms` / `791 ms`. This gives a seed-stable non-Qwen strict-prompt medium row
on Mac-local hardware; MoE/FP8 remains the only broad-architecture blocker.

Update `2026-04-28`: Gemma 4 E2B now has a direct source-signal ablation. With
the same MPS n160 seed29 setup but `raw_log_no_trace`, the private diagnostic
trace line is removed and the row collapses to matched `40/160 = 0.250`,
target-only `40/160`, best control `41/160`, packet valid rate `0.000`, and
exact-ID parity true. This strengthens the non-Qwen claim by showing the
Gemma gain is tied to the private diagnostic signal, not target priors or the
prompt wrapper alone.

Update `2026-04-28`: Gemma 4 E2B now clears a large local frozen slice.
`google/gemma-4-E2B-it` MPS trace-no-hint n500 seed29 reaches matched
`500/500 = 1.000`, target-only `125/500 = 0.250`, best source-destroying
control `126/500 = 0.252`, packet valid rate `1.000`, exact-ID parity true,
matched-minus-best-control `+0.748`, and p50 packet latency `754 ms`. This
makes Gemma the strongest non-Qwen local source-emitter row: strict prompt,
large slice, exact-ID parity, and paired source-signal ablation. The full-paper
generalization blocker remains MoE/FP8 evidence, not dense non-Qwen feasibility.

Update `2026-04-28`: Granite 3.3 2B now has a seed-stable strict-prompt boundary
and paired source-signal ablation. `ibm-granite/granite-3.3-2b-instruct` CPU
trace-no-hint n160 seed31 repeats the seed29 result exactly at matched
`101/160 = 0.631`, target-only/best control `40/160 = 0.250`, packet valid rate
`0.537`, exact-ID parity true, and p50 packet latency `3691 ms`. The paired
raw-log/no-trace n160 seed31 row collapses to matched `40/160 = 0.250`,
target-only/best control `40/160`, packet valid rate `0.000`, exact-ID parity
true, and p50 packet latency `2857 ms`. This promotes Granite from a one-off
weak non-Qwen positive to a stable prompt-contract sensitivity row, while still
keeping the full-paper claim bounded.

Update `2026-04-28`: target-model decoding is now stronger than tiny smoke.
The intended Qwen3 target-decoder n160 MPS gate failed before prediction with an
Apple MPS matmul shape error, so it is logged as backend-blocked. The CPU
fallback core seed29 n64 gate passes: matched `42/64 = 0.656`, target-only
`16/64 = 0.250`, best control `16/64 = 0.250`, valid matched predictions
`1.000`, exact-ID parity true, and p50 matched latency `2182 ms`. This reduces
the hand-coded decoder objection but remains an ablation until the held-out n64
target-decoder row also passes.

Update `2026-04-29`: systems and novelty positioning are now explicit. Added
`results/source_private_systems_summary_20260428/`, which aggregates the
deterministic rate rows, model-produced packet rows, and target-decoder rows.
The deterministic headline is: 2-byte packets reach `1.000` accuracy on core
and held-out `500`-example surfaces; matched-byte text stays at `0.250`; full
hidden-log relay also reaches `1.000` but costs `366.45-373.50` bytes, giving
`183.2x-186.7x` compression for the packet. Added
`paper/source_private_systems_novelty_review_20260429.md` and
`references/481_systems_novelty_and_future_methods_refs.md` to position the
work against C2C, KVComm, activation communication, prompt compression,
tool-agent handoff, Wyner-Ziv/indirect source coding, quantization, JEPA,
Q-Former, and diffusion-inspired successor branches. Readiness remains
scoped-positive rather than broad ICLR-ready: the current claim is defensible,
but the best next contribution is a learned syndrome packet smoke, and MoE/FP8
still remains unclaimed until endpoint access is available.

Update `2026-04-29`: the target-model decoder ablation now has paired
core/held-out CPU n64 confirmation. Held-out seed30 with `Qwen/Qwen3-0.6B` as
the target decoder reaches matched `46/64 = 0.719`, target-only `16/64 =
0.250`, best control `17/64 = 0.266`, valid matched predictions `1.000`,
exact-ID parity true, and p50 matched latency `2237 ms`. Combined with the core
n64 CPU row (`0.656` matched versus `0.250` controls), this is enough local
evidence to stop widening target-decoder rows and return to learned syndrome
packets or MoE/FP8 endpoint gates.

Update `2026-04-29`: learned syndrome packets are now the top live method
candidate beyond the hand-designed diagnostic protocol. The synthetic
source-private learned-syndrome smoke passes at the low-rate frontier on two
seeds: seed29 passes at 1/2/4 bytes with matched `0.820/0.949/0.992` versus
target `0.250`, and seed30 passes at 1/2 bytes with matched `0.797/0.902`
versus target `0.250`. Higher budgets are not promoted because source-free
binary/text controls can rise above the allowed tolerance. Readiness improves
from "protocol-only" to "has a learned-method candidate," but this is still
smoke evidence; the next exact gate is a real-feature learned syndrome row on
tool-trace or cached candidate features under the same controls.

Update `2026-04-29`: the real-feature learned syndrome row now passes on the
tool-trace/candidate-text surface. After tightening answer-masked controls to
remove `REPAIR_DIAG`, hidden input, expected/actual, failure status, test name,
and `repair_family`, the seed pair `29 -> 30` all-family train/eval `512/256`
row passes at 6 bytes with matched `0.945`, target `0.250`, best no-source
`0.285`, and full diagnostic oracle `1.000`; seed pair `31 -> 32` repeats at
6 bytes with matched `0.918`, target `0.250`, best no-source `0.289`, and full
diagnostic oracle `1.000`. This is the strongest path to a distinct method
contribution beyond diagnostic packets. The next blocker is compression-native
baselines at the same 6-byte budget: random sign sketch, rotation+scalar
quantization, and QJL/PQ-style controls.

Update `2026-04-29`: compression-native matched-byte baselines now change the
live method. The learned random-hyperplane syndrome is no longer the best
transport: a 6-byte scalar-quantized learned source projection reaches `0.979`
on a `512`-example frozen slice versus learned syndrome `0.953`, target-only
`0.250`, raw source sign sketch `0.307`, scalar shuffled `0.166`, and scalar
answer-masked `0.293`. A second `256`-example seed pair also passes strict
scalar controls with scalar `0.945` versus learned syndrome `0.910`. Readiness
improves on the systems axis because the method now has a natural quantized
packet story, but it is still not full ICLR-ready until the scalar packet
survives 5-seed repeats, held-out-family splits, codebook remapping, and
candidate-side masking. The current live branch is `scalar_quantized_source`.

Follow-up `2026-04-29`: 5-seed scalar stability is accuracy-strong but
control-incomplete. The 6-byte scalar packet has mean matched accuracy `0.972`
across five seed pairs and beats the learned sign-syndrome row on every seed
checked, but strict source-destroying controls pass only `3/5`. This is not
ready for a headline ICLR claim. The exact blocker is now control stabilization:
answer-masked and shuffled-source packets must stay near target-only across all
seeds without sacrificing the scalar packet's accuracy. A naive no-bias
source-innovation variant was pruned because it reduced matched accuracy to
`0.389` on the hard seed.

Follow-up `2026-04-29`: the live learned-method branch is now the
`slot/no-intercept` scalar packet. It narrows target-side side information to
public candidate slots and removes the ridge intercept, preventing an
answer-masked source from emitting a learned global prior packet. This gate
passes `5/5` all-family seed pairs at train/eval `768/512`: matched scalar
`1.000`, target `0.250`, constrained shuffled source `0.000`, answer-masked
`0.250`, label-shuffled ridge at or below `0.258`, and raw source sign sketch
at or below `0.307`. Readiness improves to a genuine learned-packet positive on
a scoped same-family/all-family surface. The remaining blockers for full ICLR
are cross-family generalization, codebook remap for slot identities, paired
uncertainty, and a more realistic ambiguous-candidate benchmark.

Follow-up `2026-04-29`: codebook-remap and paired uncertainty reduce one major
reviewer risk. Three remapped slot codebooks pass strict controls with scalar
accuracy `0.463-0.508` versus target-only `0.250`; paired bootstrap lower
bounds versus target-only remain positive, with the weakest lower bound
`+0.156`. The same bootstrap also shows the limitation: remap margins versus
raw source sign sketch are smaller, weakest lower bound `+0.072`. Readiness
improves for a scoped claim, but full ICLR readiness still needs either stronger
remap margins, a more realistic ambiguous-candidate benchmark, or model-emitted
packets that show the method is useful beyond this controlled slot codebook.

Follow-up `2026-04-29`: QJL/TurboQuant residual coding was tested as an opt-in
matched-byte comparator. It is clean but does not improve the remap frontier:
same-codebook remains saturated (`1.000` scalar and QJL), while remap seeds
`101/103/107` are lower for QJL by `1.6/2.3/3.5` points. This reduces the
chance that a simple residual sign sketch is the missing ICLR contribution.
Keep QJL/TurboQuant as a systems comparator and move the live method search to
relative-anchor transport or model-emitted packets.

Follow-up `2026-04-29`: relative-anchor transport now has a first positive
systems row. RASP/relative-score packets use public candidate anchors at the
source and send four quantized candidate scores. On equal 4-byte comparisons,
RASP is control-clean and improves remap mean over scalar by `+3.2` points,
with remap `101` significant versus scalar and all rows significant versus
target-only. This strengthens the paper's systems contribution and gives a
third technical mechanism, but the scalar-vs-RASP improvement is not uniformly
significant. Full ICLR readiness still needs either more remap seeds, a
cross-family RASP row, or a model-emitted packet.

Follow-up `2026-04-29`: expanded RASP evidence tightens the claim boundary.
Seven remap seeds show a positive mean equal-byte lift over scalar (`+3.7`
points), but the expanded bootstrap gate fails by the predefined rule:
minimum RASP-vs-target CI95 lower bound is `+0.146`, and one remap has a
near-threshold random-byte control failure. Cross-family remains asymmetric:
holdout-to-core passes for RASP, but core-to-holdout fails below target with
failed controls. Readiness does not improve to full ICLR yet. The next exact
method gate should canonicalize candidate order and/or train a
consistency/JEPA-style posterior packet; do not claim RASP as cross-family.

Follow-up `2026-04-29`: paired uncertainty now strengthens the live endpoint
receiver branch. The `n=64` label-strict endpoint rows pass a `5000`-sample
paired bootstrap gate across core and holdout: minimum packet-vs-target lower
CI is `+0.297`, minimum packet-vs-best-source-destroying-control lower CI is
`+0.297`, minimum strict-label packet-vs-target lower CI is `+0.281`, and
packet valid rate is `1.000`. Query-aware diagnostic text remains
accuracy-comparable, but it uses `14` bytes versus the packet's `2` bytes, so
the claim is a Pareto/rate-quality systems claim rather than an accuracy win
over all higher-byte relays. Readiness improves on the statistics/reviewer-risk
axis, but full ICLR readiness still needs the same `label_strict` gate at
`n=160`, a true server TTFT/throughput benchmark when GPU serving is available,
and a stronger learned receiver/method story beyond a hand-designed interface.

Follow-up `2026-04-29`: the first `n=160` endpoint scale-up row passes on core,
but the full rung is not complete. The core label-strict all-condition CPU run
reaches packet `0.675`, strict-label packet `0.662`, target-only `0.250`,
matched-byte text `0.250`, random same-byte `0.000`, deranged public table
`0.244`, best source-destroying control `0.250`, and valid rate `1.000`.
Paired bootstrap lower CIs are `+0.350` versus target and best control, and
`+0.338` for strict-label packet versus target. Query-aware text is slightly
higher at `0.694` but costs `14` bytes versus the packet's `2` bytes. This
raises confidence that the local signal scales beyond `n=64`, but readiness
still needs holdout `n=160` and a paired core+holdout `n=160` uncertainty
summary before calling the medium rung cleared.

Follow-up `2026-04-29`: the local medium endpoint rung is now cleared. Holdout
`n=160` label-strict passes with packet `0.688`, strict-label packet `0.675`,
target-only `0.250`, matched-byte text `0.250`, random same-byte `0.000`,
deranged public table `0.244`, best source-destroying control `0.250`, and
valid rate `1.000`. The combined core+holdout `n=160` paired uncertainty gate
passes with minimum packet-vs-target and packet-vs-best-control lower CIs of
`+0.350`, and minimum strict-label packet-vs-target lower CI of `+0.338`.
Readiness improves to medium local confirmation for the endpoint branch. The
remaining ICLR blockers are now (1) server-side TTFT/throughput on a GPU
endpoint, (2) clearer comparison/framing against C2C/KVComm/activation
communication and prompt compression, and (3) a learned receiver or
target-preserving bottleneck that reduces the hand-designed protocol concern.

Follow-up `2026-04-29`: started the learned-receiver branch and got a first
positive smoke. `scripts/run_source_private_candidate_embedding_receiver.py`
trains a source encoder plus target-side candidate scorer over public candidate
features and packet/candidate interactions, with a calibrated target-preserving
margin gate. On all-family train/eval `768/512`, seed `29 -> 30`, the 4-byte
receiver reaches matched `0.748`, target `0.250`, best destructive control
`0.262`, and full diagnostic oracle `0.998`; zero-source and shuffled-source
both stay at `0.250`. This improves the technical-contribution portfolio, but
it is still a smoke: next gates are 3-seed repeat, held-out-family split, paired
uncertainty, and comparison to scalar WZ/endpoint rows.

Follow-up `2026-04-29`: the learned receiver is now a bounded contribution with
a clear failure boundary. The 4-byte receiver is not seed-stable (`2/3` seeds
pass; one seed drops to matched `0.328` and matched-control delta `+0.049`).
At `8` bytes, the same-distribution all-family receiver passes `3/3` seeds with
matched mean `0.749`, matched minimum `0.514`, max destructive control `0.283`,
and minimum matched-control delta `+0.230`. However, the core-to-holdout
receiver fails at `8` bytes: matched `0.453`, target `0.250`, best destructive
control `0.311`, full diagnostic oracle `0.809`. Removing raw candidate
feature coordinates worsens heldout transfer (`0.332` at `n=256`). Readiness
therefore improves for the same-distribution learned-decoder contribution, but
not for cross-family learned communication. The next Mac-local method gate is a
family-invariant receiver using anchor-relative/codebook features or
fold-heldout calibration at `8` bytes.

Follow-up `2026-04-29`: tested simple family-invariant learned receiver fixes,
and all failed core-to-holdout. The coordinate-free hashed code-similarity
receiver has perfect oracle decoding (`1.000`) but target-level matched
accuracy (`0.256`), so the source encoder is not producing transferable
candidate-code packets. The naive anchor-relative code-similarity receiver
keeps controls clean but reaches only matched `0.281` with oracle `0.756`; the
anchor-relative ridge receiver reaches matched `0.303` but lets controls
dominate (`0.438` best destructive). This prunes raw code-similarity and simple
cosine-anchor banks as cross-family fixes. Readiness is unchanged: the learned
receiver remains a same-distribution contribution only. Next best method gate:
fold-heldout calibration or sparse/shared-dictionary receiver with explicit
anchor-remap/private-atom controls.

Follow-up `2026-04-29`: added a reviewer-facing pass/fail ledger derived from
the full CPU systems frontier. The artifact has `104` rows: `3` paper-ready
paired-uncertainty rows, `58` positive rows needing more evidence, `1` weak
positive, and `42` failed/pruned rows. This improves auditability and makes the
claim boundary less cherry-picked, but readiness does not become full ICLR:
the strongest paper-ready rows are still endpoint-proxy CPU evidence rather
than server TTFT/throughput, and the learned receiver remains same-distribution
only. Literature scout memo `references/494_iclr_strengthening_scout_20260429.md`
sets the next two highest-value additions: KV/cache byte lower-bound accounting
against TurboQuant/QJL/KIVI/KVQuant/SnapKV/CacheGen, then a masked or sparse
source-private innovation receiver gate.

Follow-up `2026-04-29`: added the KV/cache byte lower-bound accounting artifact
requested by the systems scout. On the `n=160` label-strict endpoint rows, the
2-byte packet is compared to cache-payload estimates for extra prompt tokens
under fp16/bf16, int8, int4, TurboQuant-style `3.5`/`2.5` bit,
KIVI/KVQuant-style `2` bit, and QJL-style `1` bit assumptions. The minimum
non-packet QJL-style cache payload is `10,752.0x` the packet; the minimum
KIVI-style cache payload is `21,504.0x` the packet. This strengthens the
systems/rate contribution and makes the comparison against KV compression more
honest. It does not yet provide production serving throughput, so full ICLR
readiness remains blocked by real GPU endpoint TTFT/TPOT/throughput and a
cross-family sparse or masked innovation receiver.

Follow-up `2026-04-29`: implemented the first sparse masked source-private
innovation receiver. Same-distribution smoke is strong: all-family train/eval
`128/64` reaches matched `0.766` at `4` bytes and `0.922` at `8` bytes, target
`0.250`, best destructive controls `0.281`/`0.266`, and oracle `1.000`.
However, the first strict cross-family direction fails decisively:
core-to-holdout `256/128` reaches only `0.258` at `4` bytes and `0.250` at
`8/12` bytes, target `0.250`, while oracle remains `1.000`. This means the
anchor-relative innovation code can represent the answer, but the learned
source-private innovation map does not transfer across held-out families.
Readiness does not improve; the branch is not promoted. Next method gate must
change the representation or supervision, likely shared-dictionary/crosscoder
calibration with feature knockout, before rerunning cross-family.

Follow-up `2026-04-29`: tested a stricter shared-text semantic view for the
masked innovation receiver. Candidate features use shared lexical intent/public
issue features and mask diagnostic handles, so this is less protocol-coded than
the anchor-relative view. It still fails core-to-holdout: matched `0.266` at
`4` bytes and `0.250` at `8` bytes versus target `0.250`; all destructive
controls stay at target and oracle stays `1.000`. This is the second adjacent
variant failing for the same reason. Do not tune simple masked-innovation
projection further. The only live learned cross-family follow-up is a materially
different shared-dictionary/crosscoder model with feature knockout, or else the
paper should keep learned receivers as same-distribution diagnostics only.

Follow-up `2026-04-29`: the coded-label/protocol risk gate passes and
materially strengthens the current scoped positive-method claim. On `160`
frozen examples, three code seeds, and five transform rows (`baseline`,
opaque-label rename, diagnostic-code remap, candidate-order permutation, and
the composed label+code+order stress), the 2-byte packet remains at `1.000`
accuracy while target-only is `0.250`, reviewer-negative controls are `0.250`,
and the worst source-destroying control is `0.263`. This weakens fixed-label,
fixed-order, and single-codebook lookup objections. Readiness improves for the
source-private evidence-packet paper, but not to full ICLR-ready status:
remaining gaps are one-command reproduction/novelty packaging, real serving
TTFT/TPOT/throughput, and either a broader cross-family learned method or an
explicitly scoped claim that reports cross-family failures honestly.

Follow-up `2026-04-29`: added the ICLR evidence bundle, which consolidates the
current contribution stack into one machine-checkable artifact. The bundle
passes `10/10` checks and reports `5` contribution rows: source-private
benchmark/controls, extreme-rate candidate-syndrome packets, systems
byte/KV-cache accounting, endpoint paired uncertainty, and learned-receiver
diagnostics with explicit failure boundaries. It also includes an `8`-row
novelty matrix against C2C, KVComm, TurboQuant/QJL, prompt compression,
source-coding theory, and JEPA/diffusion latent prediction. Readiness improves
for review packaging and reproducibility. It remains scoped rather than fully
ICLR-secure because production serving telemetry and broad cross-family learned
communication are still missing. The next exact gate is a negative-boundary
appendix over cross-family failures and oracle headroom, followed by either an
`n=500` composed coded-label stress or paper revision.

Follow-up `2026-04-29`: added the cross-family negative-boundary appendix. It
aggregates `27` rows across `6` method families and records `0` claim-ready
cross-family methods, while preserving `6` oracle-headroom rows. This improves
readiness by making the paper's limitation credible rather than hidden: learned
WZ, canonical RASP, consistency posterior, anchor-relative sparse packets,
candidate-embedding receivers, and masked innovation receivers all fail or are
asymmetric under strict cross-family controls. The next method branch, if any,
must be materially different: shared sparse dictionary/crosscoder packets with
feature knockout and bidirectional controls. Otherwise, proceed to paper
revision around the scoped source-private evidence-packet claim.
