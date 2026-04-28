# Source-Private Literature-To-Method Sprint

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; roughly one source-private strict-small
   pass plus medium confirmation, seed stability, and cross-family replication
   away from a paper claim.
2. Current paper story: ordinary SVAMP/GSM residual hunting is saturated by
   target priors, prompt wrappers, and answer leakage. The new story is
   source-private residual communication: source observes `S`, target has
   public prompt/candidate/cache side information `X,T`, source sends
   rate-capped `M`, target decodes with `X,T,M`.
3. Exact blocker to submission: no strict-small benchmark yet proves that a
   source-derived message improves the target only when matched private source
   evidence is present.
4. Current live branches: private evidence packet / candidate-syndrome side
   information; private tool-trace handoff; source-private Query-JEPA/Q-Former
   adapters only after a residual surface exists.
5. Highest-priority gate: implement and run the cheapest strict source-private
   evidence-packet gate with zero/shuffled/random/answer-only/answer-masked and
   matched-byte text controls.
6. Scale-up rung: smoke-to-strict-small benchmark contract.

## Committee Synthesis

The subagents converged on the same deadline strategy from different
literatures:

- JEPA/V-JEPA/LeJEPA: useful for anti-collapse and latent prediction, but too
  risky as the first one-month method. Use detached target candidate/process
  latents, source-innovation scores, variance/effective-rank/covariance/query
  entropy, and dual-view agreement only after the benchmark has residual IDs.
- VLM connectors: Q-Former, Perceiver Resampler, Flamingo, BLIP-2, LLaVA, and
  LLaMA-Adapter justify fixed query bottlenecks, projectors, and zero-init
  target-preserving gates. They also warn that simple projectors and prompt
  formats are strong baselines.
- KV/compression: KVComm/KVTC/KIVI-style baselines make byte and latency
  accounting mandatory. Cache transfer is a comparator, not the first claim,
  unless it beats source-destroying controls and text at a defensible byte
  budget.
- Information theory: the right framing is Slepian-Wolf/Wyner-Ziv/distributed
  indirect source coding. The source should send residual bits or a syndrome
  decoded using target-side candidates, not a full explanation.
- Multi-agent/tool systems: private tool traces, hidden logs, RAG shards, and
  memory handoffs are strong source-private settings. Tool-log handoff is a
  good second benchmark if the evidence-packet path passes.
- Reviewer: the one-month paper gets rejected if it leads with Query-JEPA,
  lacks target wrapper/no-source/text baselines, or lets the source packet copy
  final answers.

## Ranked Method Portfolio

### 1. Private Evidence Packet / Candidate-Syndrome Decoder

Task: target sees a public question plus candidate side information; source
sees private evidence selecting the correct candidate. Source sends a fixed-byte
syndrome or evidence packet that the target decodes against its candidate pool.

Method: keyed or learned candidate syndrome over answer-masked/private evidence
features, optionally anchor-relative to target candidates. Decoder ranks target
candidates with `X,T,M`.

Baselines:

- target-only prior
- target prompt-wrapper/no-source candidate baseline
- no-source candidate-pool recall/oracle split from selector accuracy
- structured text relay at matched bytes
- full structured text/full evidence oracle
- source-final-only and answer-only packets

Controls:

- zero-source
- deterministic shuffled-source nonself
- random same-byte packet
- answer-only packet
- answer-masked packet
- target-derived packet

Strict-small pass:

- `100-200` frozen examples.
- matched source beats best no-source by `>=15` points.
- zero/shuffled/random/answer-only/answer-masked/target-only sidecars stay
  within `2` points of best no-source.
- matched-byte structured text stays near no-source or the syndrome gives a
  clear byte/latency tradeoff against full structured text.
- report bytes, latency, candidate-pool recall, and control clean union.

Why this can be an ICLR paper: it gives a clean rate-distortion communication
claim under decoder side information, with interpretable messages and direct
source-destroying controls.

### 2. Private Tool-Trace / Test-Log Handoff

Task: source has tool observations, hidden tests, SQL/API results, or logs;
target has the task and candidate actions but not the private observations.

Method: source sends a compact answer-masked diagnostic packet or trace
syndrome; target selects the action/patch/answer.

Baselines and controls: target-only, full-log oracle, structured trace relay,
compressed text relay, source-final-only, answer/patch-masked trace, shuffled
trace, zero trace, random same-byte trace.

Strict-small pass: `100` tasks, `>=20` point gain over best target-prior,
source-destroying controls within `2-5` points of target-prior, full-log oracle
above target-prior to prove headroom.

Why this can be an ICLR paper: it is real agent-to-agent communication with
systems value: compact diagnostics instead of full logs/tool outputs.

### 3. Source-Private Query-JEPA / Zero-Init Q-Gate

Task: run only on source-private residual IDs where target candidate pools
contain gold but no-source selectors fail.

Method: learned queries cross-attend to source-private masked state, predict
detached target candidate/process latents, and inject through a zero-init gate.
Score source innovation as matched source minus zero-source/target-prior score.

Baselines and controls: target-prior latent scorer, slots-only query bank,
target-only adapter, zero-source, shuffled-source, random same-byte vector,
wrong challenge hash, structured text.

Strict-small pass: matched source beats best control by `>=15` points, all
controls near no-source, step-zero target output identical, no target-correct
harm, effective rank/variance/query entropy healthy.

Why second-stage: it has higher novelty but repo history says connector-first
branches repeatedly fail source controls.

### 4. Anchor-Relative Sparse Evidence Codes

Task: same as private evidence packet, but source message is an anchor-relative
rank/sign code instead of a direct hash/syndrome.

Method: source encodes private evidence by similarities/ranks to target
candidates or shared anchors. Decoder uses relative geometry to rank candidates.

Baselines and controls: same as candidate-syndrome plus raw-coordinate
projector, shuffled anchors, target-only anchors, and CKA/SVCCA diagnostics.

Strict-small pass: recover residual IDs with `<=16` bytes/example and no
control clean union.

Why lower priority: more interpretable than raw latents, but slower to validate
than the direct evidence-packet gate.

### 5. KV/Cache Source-Private Transport Baseline

Task: source has private evidence context; target receives selected/compressed
source KV/cache state.

Method: reuse KVComm controls and add structured text relay at matched bytes,
plus low-bit simulated bytes from KVTC/KIVI-style accounting.

Strict-small pass: at least one matched-only win where target-only, zero-source,
shuffled-source, and structured text fail, with source-answer overlap false.

Why baseline not headline: current KVComm smoke communicates hundreds of KB per
example, so it needs either strong accuracy or a compression breakthrough.

## Decision

The single highest-probability one-month path is:

1. Build a source-private evidence-packet/candidate-syndrome benchmark contract.
2. Run a deterministic smoke to verify controls, byte accounting, and
   pass/fail machinery.
3. Replace the deterministic source/target with LLM target candidate generation
   and source packet generation on `100-200` private evidence examples.
4. Only if this exposes stable residual IDs, train a Query-JEPA/Q-Former
   source-private connector as the learned method.

## Smoke Gate Run

Implemented `scripts/run_source_private_evidence_packet_gate.py`.

Command:

```bash
./venv_arm64/bin/python scripts/run_source_private_evidence_packet_gate.py \
  --examples 128 \
  --candidates 4 \
  --seed 17 \
  --syndrome-bytes 2 \
  --output-dir results/source_private_evidence_packet_gate_20260427
```

Result:

- target-only: `32/128`, accuracy `0.250`
- target-wrapper/no-source: `32/128`, accuracy `0.250`
- matched 2-byte syndrome: `128/128`, accuracy `1.000`
- zero-source: `32/128`, accuracy `0.250`
- shuffled-source: `32/128`, accuracy `0.250`
- random same-byte: `32/128`, accuracy `0.250`
- answer-only: `32/128`, accuracy `0.250`
- answer-masked: `32/128`, accuracy `0.250`
- target-only sidecar: `32/128`, accuracy `0.250`
- structured text at matched 2 bytes: `32/128`, accuracy `0.250`
- full structured text: `128/128`, accuracy `1.000`, mean `13` bytes
- matched minus best no-source/control: `+0.750`
- gate: `pass`

Interpretation: this is not paper evidence yet because it is deterministic and
synthetic. It is a benchmark/harness contract proving that the pass/fail
machinery distinguishes a matched source-private syndrome from target priors,
same-byte random/source-destroying controls, and matched-byte text relay. The
next decisive gate must instantiate the same contract with LLM-generated target
candidates and source packets on frozen private evidence examples.

## Next Exact Gate

Build `source_private_evidence_packet_strict_small_20260428`:

- `100-200` synthetic-but-natural private evidence QA examples with held-out
  entities/templates.
- Target model sees question plus candidate pool and produces `S32` no-source
  candidates.
- Source model sees private evidence and emits a capped packet at `2`, `4`,
  `8`, `16`, and `32` bytes/tokens.
- Decoder evaluates candidate selection with exact ID parity.
- Include target-only, target wrapper, no-source candidate pool, full evidence,
  structured text at matched bytes, full structured text, zero-source,
  shuffled-source, random same-byte, answer-only, answer-masked, and
  target-derived sidecar controls.
- Pass if matched packet beats best no-source by `>=15` points and every
  source-destroying control stays within `2` points of no-source.

If strict-small passes, the next learned method is source-private Query-JEPA /
zero-init Q-gate over the same frozen IDs.
