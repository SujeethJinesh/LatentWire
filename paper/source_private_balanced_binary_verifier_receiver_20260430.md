# Balanced Binary-Verifier Receiver Gate

- date: `2026-04-30`
- artifacts:
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_choice_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_binary_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n64_binary_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_n64_binary_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_n64_binary_logprob_cpu_2seed/`
- status: strict-small frozen Qwen receiver passes with a calibrated binary
  verifier; choice-token likelihood is pruned

## Cycle Start

1. Current ICLR readiness and distance: stronger scoped positive-method paper,
   still not comfortably ICLR-ready. Distance is now one less protocol-shaped
   receiver or native serving telemetry, not another tiny diagnostic smoke.
2. Current story: the source privately observes hidden evidence, sends a
   2-byte diagnostic packet, and the target uses public candidate side
   information to resolve the correct candidate. Gains must vanish when the
   source signal is zeroed, shuffled, random, truncated text, or public-only.
3. Exact blocker: reviewers can still object that the best direct result is a
   hand-decoded table lookup unless a frozen or learned receiver consumes the
   packet under balanced controls.
4. Current live branch: balanced diagnostic packet plus constrained target-side
   receiver selectors.
5. Highest-priority gate: strict-small frozen Qwen3 receiver with validity,
   source-destroying controls, paired uncertainty, and seed stability.
6. Scale-up rung: strict small receiver gate.

## Layman Version

The target sees four possible fixes and a short public code attached to each
fix. The source privately knows which hidden test failed and sends only a
two-character clue. Instead of using a hand-written rule, I asked a small Qwen
model a simpler question four times: "does this clue match this candidate?"
If Qwen says yes for one candidate, the target picks it; if Qwen says no for
all candidates, the target falls back to its own prior guess. This checks
whether a real language model can use the packet without free-form output
formatting artifacts.

## Harness Change

`scripts/run_source_private_tool_trace_target_decoder_smoke.py` now supports
two constrained likelihood-style decoder modes:

- `choice_logprob`: scores the next-token likelihood of `A/B/C/D` under the
  choice-alias prompt. This is diagnostic only because it is a one-token
  option-prior probe, not full continuation likelihood.
- `candidate_binary_logprob`: asks a yes/no equality question for each
  candidate and scores the next-token `yes` versus `no` margin.

The binary verifier uses a calibrated fallback rule: only a positive
`yes_minus_no` margin is allowed to override the target prior. The first n4
smoke without this threshold failed because random packets selected the "least
no" candidate. After the threshold, shuffled/random/no-source conditions fall
back to target prior.

Tests added cover packet validity filtering, answer-label exclusion in the
binary prompt, token-surface scoring, binary fallback behavior, and positive
match selection.

## Negative: Choice-Token Likelihood

Command:

```bash
env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_diag_only_public_ablation_20260430/direct_diag_n500_seed29/benchmark.jsonl \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_choice_logprob_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --limit 32 --seed 29 --max-new-tokens 1 --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte \
  --prompt-mode choice_alias --decode-mode choice_logprob
```

Outcome:

- pass gate: `False`
- matched packet: `0.250`
- target-only: `0.250`
- best control: `0.250`
- valid prediction rate: `1.000`
- interpretation: forced option-token likelihood collapses to option priors and
  does not use the packet.

Decision: prune choice-token likelihood as a receiver method. Keep it as a
label-prior diagnostic.

## Positive: Calibrated Binary Verifier

Seed 29 n32:

- artifact:
  `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n32_binary_logprob_cpu/`
- pass gate: `True`
- matched packet: `32/32 = 1.000`
- target-only: `8/32 = 0.250`
- shuffled/random/truncated-text controls: `0.250`
- valid prediction rate: `1.000`
- matched p50 CPU latency: `1131.4 ms`
- paired CI95 low vs target and best control: `+0.594`

Seed 29 n64:

- artifact:
  `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_n64_binary_logprob_cpu/`
- pass gate: `True`
- matched packet: `64/64 = 1.000`
- target-only: `16/64 = 0.250`
- shuffled/random/truncated-text controls: `0.250`
- valid prediction rate: `1.000`
- matched p50 CPU latency: `1087.6 ms`
- paired CI95 low vs target and best control: `+0.641`

Seed 31 n64:

- artifact:
  `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_n64_binary_logprob_cpu/`
- pass gate: `True`
- matched packet: `64/64 = 1.000`
- target-only: `16/64 = 0.250`
- shuffled packet and truncated-text controls: `0.250`
- random same-byte: `17/64 = 0.266`
- valid prediction rate: `1.000`
- matched p50 CPU latency: `1119.1 ms`

Combined two-seed paired uncertainty:

- artifact:
  `results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_n64_binary_logprob_cpu_2seed/`
- rows: `2`
- pass rows: `2`
- min matched-target: `+0.750`
- min matched-best-control: `+0.734`
- min CI95 low vs target: `+0.641`
- min CI95 low vs best control: `+0.625`
- min valid prediction rate: `1.000`

## Interpretation

This is now a stronger receiver defense than the earlier label-output prompt:
validity is perfect, source-destroying controls collapse to the target prior,
and the result repeats over two seeds at n64. The model is not free-generating
candidate labels; it is acting as a calibrated target-side verifier over public
candidate side information.

This does not prove broad cross-model latent transfer. The receiver still has
an explicit public side-information contract. The safe claim is:

> A frozen LLM can mediate a source-private 2-byte diagnostic packet into a
> target-side candidate decision under balanced source-destroying controls.

## Systems Read

The systems win remains bytes and boundary exposure, not CPU latency. The
binary verifier uses a 2-byte source packet and no source text or source KV
state, but the Mac CPU implementation runs four target forward passes per
condition and has about `1.1 s` p50 matched latency. A production systems claim
requires shared-prefix batching or structured decoding in a native serving stack
with TTFT/TPOT/goodput and memory counters.

## Decision

Promote:

- calibrated binary-verifier receiver as strict-small supporting evidence for
  model-mediated packet consumption;
- two-seed n64 paired uncertainty as the cleanest current receiver row;
- choice-token likelihood as a negative label-prior diagnostic.

Do not promote:

- broad latent-transfer claims;
- production latency claims;
- choice-token receiver as a method.

## Next Exact Gate

`source_private_balanced_binary_verifier_receiver_cross_family_20260501`

Run the same binary verifier on `direct_core_n500_seed29` and
`direct_holdout_n500_seed29` at `n=64`, then one `n=160` row if both pass.
Pass rule: matched packet beats target and best control by `>= +0.15`, valid
rate is `>= 0.95`, paired CI95 lower bound is `> +0.10`, and random same-byte
stays within target `+0.05`.

## Tests

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py

./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_target_decoder_smoke.py \
  tests/test_summarize_source_private_target_decoder_uncertainty.py
```

Outcome: `17 passed in 0.14s`.
