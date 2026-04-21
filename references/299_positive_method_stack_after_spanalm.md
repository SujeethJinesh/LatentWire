# Positive Method Stack After Span-ALM Negative

Scope: the current cross-tokenizer / teacher-side branch after the span-ALM
and span-smoothed likelihood teacher both failed to rescue the controlled
GSM slice.

Working read:
- direct next-token likelihood mass is too brittle
- span-smoothed likelihood is still too weak
- the next positive branch should shift toward attention / interaction
  supervision and only then widen to tokenizer-remap methods

## Ranked next 5 method attempts

| Rank | Method attempt | Compose with current branch? | Cheap? | New code? | Why it is next |
|---|---|---|---|---|---|
| 1 | **Attention / interaction readout teacher** (`bridge_ridge_qk_readout_adapter`, `bridge_ridge_qk_dynalign_interact_module_replace`) | Yes. Stacks directly on the current `spanalign` / `ctxalign` / `dynalign` plumbing. | **Yes** | **No / minimal** | The current failure says direct likelihood is wrong; the next useful teacher is the one that matches prompt-local attention or interaction structure instead of token mass. |
| 2 | **Cross-tokenizer preference distillation on aligned spans** (CTPD-style) | Yes. Uses the same aligned-span machinery but changes the supervision target. | **Medium** | **Yes** | This is the cleanest span-level teacher that still survives tokenizer mismatch while moving away from exact next-token boosting. |
| 3 | **Contextual dynamical mapping + soft alignment** (dynamic remap before the bridge) | Yes. Best as a routing/alignment layer feeding the teacher above. | **Medium** | **Yes** | The negative result suggests the correspondence itself must move with context, not stay fixed across prompts. |
| 4 | **Attention-aware token initialization / token distillation** | Partially. Helps the target-side interface once span supervision is working. | **Medium** | **Yes** | If the bridge is still brittle, initializing new tokens from attention-aware teacher states gives a cheaper tokenizer-side repair than full vocab surgery. |
| 5 | **Tokenizer adaptation / token alignment** (TokAlign-style) | Yes, but only as a later-stage compatibility layer. | **Low** | **Yes** | This is the heaviest way to remove tokenizer mismatch. It is more powerful than span remapping, but too expensive to be the next step. |

## What to do with each

### 1) Attention / interaction readout teacher

Best role:
- first positive branch after span-ALM failure
- reuses the current bridge, but changes the teacher from likelihood mass to
  readout / interaction structure

Why this composes:
- works on top of the current geometry and routing stack
- does not require tokenizer surgery
- can be evaluated against the same GSM / SVAMP readouts

### 2) CTPD-style aligned-span preference distillation

Best role:
- span-level teacher that is still tokenizer-agnostic
- stronger than the span-ALM teacher because the target is preference / span
  structure rather than exact token likelihood

Why this composes:
- uses the same span alignment layer
- can sit under the readout teacher as a secondary loss

### 3) Contextual dynamical mapping + soft alignment

Best role:
- fix the correspondence layer before supervision
- use dynamic span remapping when token boundaries drift

Why this composes:
- can feed both the readout teacher and the span-preference teacher
- is the best way to make the teacher objective less token-bound

### 4) Attention-aware token initialization

Best role:
- small tokenizer-side repair once a better span teacher exists
- useful when supervision wants to survive boundary drift but not full vocab
  replacement

Why this composes:
- plugs into the same teacher signal
- can be tested as a target-side adapter rather than a new transport map

### 5) TokAlign / token alignment

Best role:
- last-resort tokenizer bridge when the pair gets more heterogeneous

Why this composes:
- helps all of the above, but should not be the next experiment
- too expensive for the current positive-method push

## Minimal execution order

1. keep the existing span alignment harness
2. swap likelihood mass for attention / interaction readout supervision
3. add CTPD-style span preference distillation
4. only then add dynamic remapping as the correspondence layer
5. use token-init or TokAlign only if the tokenizer mismatch remains the bottleneck

## Practical summary

After the span-ALM negative, the next positive-method attempt should be a
**readout / interaction teacher on aligned spans**, not another direct
likelihood-mass variant. If that still fails, move immediately to
CTPD-style span preference distillation and dynamic remapping before doing any
tokenizer surgery.
