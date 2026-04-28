# Source-Private Final-Review References

Date: `2026-04-30`

## Blocker Helped

The compiled paper's core evidence is strong for the scoped protocol claim, but
reviewers could reject it as an under-framed coded-label toy. This reference
patch strengthens the framing around decoder-side information, multi-agent
handoff, prompt-compression baselines, latent/KV communication competitors, and
test-guided repair scope.

## Sources And Experiment Implications

- Distributed indirect source coding with decoder side information,
  https://arxiv.org/abs/2405.13483
  - Helps: formalizes cases where the source observes evidence about a task
    latent rather than a fully reconstructable target state.
  - Mechanism/design idea: task distortion under decoder-side information.
  - Next experiment change: none for this gate; supports the source-private
    evidence-packet framing.
  - Role: theory support and paper framing.
- AutoGen, https://arxiv.org/abs/2308.08155
  - Helps: grounds the motivation in multi-agent handoff systems.
  - Mechanism/design idea: specialized agents communicate through explicit
    conversation/workflow contracts.
  - Next experiment change: future real-agent benchmark should use source-only
    tool observations with the same source-destroying controls.
  - Role: paper framing.
- LLMLingua, https://aclanthology.org/2023.emnlp-main.825/
  - Helps: addresses reviewer expectation for prompt/context compression
    baselines.
  - Mechanism/design idea: compress visible prompt tokens to reduce inference
    cost.
  - Next experiment change: none for the source-private setting; keep
    matched-byte text/JSON/full-log relay rows as the relevant controls.
  - Role: baseline framing.
- C2C, https://arxiv.org/abs/2510.03215, and KVComm,
  https://arxiv.org/abs/2510.03346
  - Helps: names the closest latent/cache communication competitors.
  - Mechanism/design idea: communicate internal model state rather than an
    explicit interpretable evidence packet.
  - Next experiment change: future learned-interface work should compare to
    C2C/KVComm-style cache transfer under matched source-private controls.
  - Role: baseline and limitation framing.
- Repair-R1, https://arxiv.org/abs/2508.13179
  - Helps: keeps the hidden-repair benchmark connected to test-guided program
    repair while preserving the paper's candidate-selection boundary.
  - Mechanism/design idea: test feedback as repair evidence.
  - Next experiment change: real repair workflows are future work; do not claim
    unconstrained repair from this benchmark.
  - Role: benchmark framing and limitation support.

## Consequence For The Paper

The final-review edit should not add a new headline experiment. It should
tighten the manuscript around a scoped claim: explicit source-private
diagnostic packets are useful in the far-left-rate regime when the target has
candidate-side metadata as decoder side information. Learned latent transfer,
general program repair, and real multi-agent tool workflows remain future
comparisons.
