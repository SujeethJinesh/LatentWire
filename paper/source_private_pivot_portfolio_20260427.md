# Source-Private Communication Pivot Portfolio

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready, but a one-month positive method is still
   plausible if we pivot away from ordinary math residual hunting.
2. Current paper story: SVAMP/GSM arithmetic is dominated by target priors,
   prompt-wrapper candidate search, and source-answer leakage. The stronger
   story is source-private residual communication under decoder side
   information.
3. Exact blocker to submission: demonstrate a source-derived, rate-capped
   message that improves a target beyond target-only/prompt-wrapper/no-source
   priors and collapses under source-destroying controls.
4. Current live branches: source-private candidate-syndrome sidecars;
   C2C/KV-teacher distillation into compact source messages; source-private
   Query-JEPA/adapter connectors only after a residual surface exists.
5. Highest-priority gate: build a strict small source-private benchmark where
   the source truly has information the target lacks.
6. Scale-up rung: strict small gate design.

## Why Pivot

The current loop is failing for a structural reason: in ordinary SVAMP/GSM, the
source usually does not have reliable private information, and the target can
rediscover many answers through sampling and prompt wrappers. That makes every
candidate method compete against a moving target-prior frontier and turns
positive-looking IDs into prompt-format artifacts.

The one-month path is to make the communication problem explicit:

- target has public prompt `X` and its own side information `T`
- source has private observation/state `S`
- source sends bounded message `M`
- target predicts with `X, T, M`
- gains must disappear when `S` is zeroed, shuffled, answer-masked, or replaced
  with random same-byte side information

This changes the paper from “latent alignment sometimes helps math” to
“source-private residual information can be compressed and decoded by another
LLM/agent under strict controls.”

## Ranked Paper Ideas

### 1. Private Evidence Packet For Retrieval QA

Core method: source sees a private evidence passage or document shard; target
sees only the question and target-side candidate pool. Source sends a compact
packet: candidate-elimination syndrome bits, relation/entity predicates, or a
small learned latent/resampler output.

Why source has private information: the answer requires a fact or relation that
is absent from the target prompt and only present in the source-visible shard.

Strict small gate:

- generate or materialize `100-200` QA items with private evidence
- target wrapper `S32` and no-source candidate oracle must be near chance or
  clearly below source-private method
- source message budget: `16-64` bytes or `16-64` tokens
- pass if matched source beats best target-prior/no-source control by `>=15`
  points and shuffled/zero/random sidecars stay near target-only

Baselines and controls:

- target-only, target prompt-wrapper `S32`, no-source candidate oracle
- full passage relay, compressed text relay, structured JSON evidence relay
- source final-only, answer/entity-masked packet, zero-source, shuffled-source,
  random same-byte packet

ICLR value: clear benchmark contribution plus rate-distortion curves for
source-private evidence transfer. Interpretable packets can expose preserved
entities, relations, and candidate vetoes.

Risk: a text packet may become extractive answer copying. Mitigation: answer
masking, entity masking, held-out evidence templates, and explicit final-answer
leak audits.

### 2. Private Tool-Trace Distillation

Core method: source has asymmetric tool access: calculator, SQL, unit tests,
retriever, simulator, or API. Target cannot call the tool. Source sends a
compact process trace, syndrome, or latent summary; target uses it to solve.

Why source has private information: the source observes tool outputs or runtime
state unavailable to the target.

Strict small gate:

- `100` tool-asymmetric tasks
- source sends answer-masked top-k trace predicates or learned latent code
- pass if matched source improves target by `>=20` points and zero/shuffled
  traces fail

Baselines and controls:

- target-only, target wrapper, full tool-output oracle
- text trace relay, structured trace relay at matched bytes
- final-answer masked trace, shuffled trace, zero trace, random trace,
  source-final-only

ICLR value: practical agent-to-agent communication under asymmetric tool access.
Systems value is direct: transmit compact diagnostics instead of full logs/tool
outputs.

Risk: trace contains final answer. Mitigation: tasks where tool outputs are
intermediate constraints rather than final answers, plus answer masking and
source-final-only controls.

### 3. Candidate-Syndrome Side Information

Core method: target generates candidate answers/rationales; source sends a
low-rate syndrome over answer-masked private information that eliminates wrong
candidates. Decoder uses target side information plus the syndrome.

Why source has private information: source observes private evidence, tool
state, hidden table rows, or C2C/KV teacher signals unavailable to target.

Strict small gate:

- target candidate pool `K=8-32`
- source message `1-8` bytes
- pass if matched syndrome recovers `>=3/8` strict small residual IDs while
  all source-destroying controls recover `0`

Baselines and controls:

- target candidate oracle and no-source merged pool
- text relay at same byte budget
- answer-only syndrome, shuffled syndrome, zero syndrome, random same-byte
  syndrome, target-derived syndrome

ICLR value: interpretable connection to Slepian-Wolf/Wyner-Ziv decoder-side
information. The message is not a prompt; it is a rate-limited residual code.

Risk: if candidate pool lacks the gold answer, syndrome cannot help. Mitigation:
separate candidate-generation oracle from selector accuracy, and report both.

### 4. Source-Private Query-JEPA Adapter

Core method: a small query bank cross-attends to answer-masked source-private
state and predicts frozen target candidate/process latents. The score is
target-prior-subtracted:

```text
source_innovation_score = score(source_adapter, candidate) - score(target_prior, candidate)
```

Why source has private information: source input includes private evidence/tool
state that target prompt-wrapper controls cannot access.

Strict small gate:

- run only after source-private candidate-syndrome surface has residual IDs
- pass if matched source recovers residual candidates while zero/shuffled/
  answer-only/slots-only/random controls recover none

Baselines and controls:

- target-prior latent scorer, target-only adapter, slots-only query bank,
  shuffled-source adapter, zero-source adapter, random same-byte source vector

Anti-collapse telemetry:

- finite score coverage
- effective rank
- variance floor
- covariance off-diagonal mass
- query entropy and dead-query rate
- dual answer-masked view agreement vs shuffled agreement
- matched-vs-best-control margin

ICLR value: if positive, this gives a mechanistic latent connector with
anti-collapse diagnostics, not just a hand-coded sidecar.

Risk: learned connector overfits quickly. Mitigation: freeze thresholds,
calibrate on disjoint IDs, and require cross-family replication only after
strict small pass.

### 5. Private Memory Handoff

Core method: source agent reads private conversation history, user constraints,
or task state; target sees only the current request. Source sends a compact
handoff state, preferably structured constraints or a latent memory packet.

Why source has private information: hidden constraints/preferences are
randomized and absent from target prompt.

Strict small gate:

- `150` synthetic handoff tasks with hidden constraints
- pass if compact handoff improves constraint satisfaction by `>=20` points and
  shuffled/irrelevant memory fails

Baselines and controls:

- target-only, full-memory oracle, template summary, structured text summary,
  shuffled memory, irrelevant memory, random same-byte memory, answer-masked
  memory

ICLR value: agent handoff is an immediately relevant systems setting. The paper
can report bytes, latency, and privacy/utility tradeoffs.

Risk: can look like summarization. Mitigation: random hidden constraints,
fixed byte budgets, and controls that separate content from template.

### 6. Private Program-State Debugging

Core method: source sees failing tests/logs/runtime trace; target sees code and
issue text but not the private failure state. Source sends a compact diagnostic
or latent packet; target fixes the bug.

Why source has private information: failing test output and stack/runtime state
are private to the source.

Strict small gate:

- `50-100` templated Python bug tasks
- target patch is evaluated by hidden tests
- pass if compact source diagnostics improve fix rate by `>=15` points and
  shuffled diagnostics fail

Baselines and controls:

- target-only, full log oracle, structured log summary, static lint baseline,
  shuffled diagnostics, zero diagnostics, random same-byte diagnostics,
  answer/patch-masked diagnostics

ICLR value: strong systems story around source-private diagnostic transfer.

Risk: patch evaluation is more engineering-heavy. Mitigation: start with
template-generated deterministic pytest tasks.

## Deadline Plan

Week 1:

- build source-private retrieval/tool-trace strict small benchmark
- implement candidate-syndrome packet and no-source/text/structured baselines
- run `100-200` examples, one seed, full controls

Week 2:

- scale best method to `500` frozen examples
- add byte, latency, generated-token, TTFT, and source-control telemetry
- run `3` seeds if compute allows

Week 3:

- add C2C/KVComm/activation communication and compression baselines where fair
- run one cross-family pair
- add Query-JEPA adapter only if candidate-syndrome has residual IDs

Week 4:

- freeze tables, paired bootstrap, ablations, interpretability diagnostics,
  paper draft

## Immediate Next Gate

Do not train a connector yet. Build the strict-small source-private evidence or
tool-trace benchmark and run a one-byte/few-byte candidate-syndrome sidecar with
full controls.

Pass bar for the first serious gate:

- target wrapper `S32` and no-source oracle cannot solve the private task
- matched source sidecar beats best no-source baseline by `>=15` points
- shuffled/zero/random/answer-only controls are within `2` points of no-source
- structured text at matched bytes is included
- source-final/answer leakage audit is clean
- bytes and latency are reported

## Committee Bottom Line

The strongest ICLR bet is not another alignment variant. It is a rate-limited
source-private residual channel where the target already has decoder side
information. The method should start simple and interpretable: candidate
syndrome or private evidence packet. Learned latent adapters become valuable
only after the benchmark exposes stable source-private residual IDs.
