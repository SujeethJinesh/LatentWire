# 369 Competitor Benchmark and Harness Directions

Date: 2026-04-21

Scope: runnable evaluation harnesses, benchmark orchestration, direct-communication competitor repos worth cloning, and a clean way to build fair matched tables for LatentWire without contaminating the method story.

## What to clone first

Primary harnesses:

- `lm-evaluation-harness`: https://github.com/EleutherAI/lm-evaluation-harness
- `OpenCompass`: https://github.com/open-compass/opencompass

Optional second harness for multimodal or broader eval shape:

- `lmms-eval`: https://github.com/EvolvingLMMs-Lab/lmms-eval

Direct-communication / latent-communication competitors:

- `LatentMAS`: https://github.com/Gen-Verse/LatentMAS
- `Cache-to-Cache (C2C)`: https://github.com/thu-nics/C2C

## Recommended clone layout

Keep all third-party code under `references/repos/` and treat it as read-only vendor state.

Suggested layout:

- `references/repos/lm-evaluation-harness`
- `references/repos/opencompass`
- `references/repos/lmms-eval`
- `references/repos/LatentMAS`
- `references/repos/C2C`

If a repo is only needed for paper inspection, cloning is enough. Do not import its code into the main method path.

## What each harness should be used for

- `lm-evaluation-harness`: the default smoke-test harness for exact-match / multiple-choice style rows, prompt consistency, parser consistency, and lightweight model-orchestration sanity checks.
- `OpenCompass`: the second orchestration shape for broader benchmark coverage and to catch harness-specific assumptions before paper tables harden.
- `lmms-eval`: only if we need a multimodal-shaped evaluation contract or a second reproducible benchmark runner with stronger logging.
- `LatentMAS`: the direct latent multi-agent competitor for same-family latent communication and token-vs-latent budget comparisons.
- `C2C`: the direct KV-cache communication competitor for cache-to-cache transport and projector-style latent transfer.

## Fair matched-table rules

Do not mix benchmark discovery with method claims. Keep a separate benchmark track and only summarize it after the row contract is fixed.

Matched rows should hold these fixed:

- same dataset split or same cached example list
- same backbone family where the comparison requires it
- same answer parser and normalization
- same max-new-tokens / latent-step budget
- same seed policy and decoding temperature policy
- same telemetry schema for predictions, traces, and failures

Minimal telemetry to store for every row:

- `model`, `backend`, `dtype`, `device`
- `task`, `split`, `n_examples`, `seed`
- `accuracy`, `exact_match`, or task-specific score
- `input_tokens`, `output_tokens`, `latency_sec`
- `latent_steps` or `cache_budget`
- `route_help`, `route_harm`, `failure_tag`
- `parser_version`, `prompt_hash`, `output_hash`

## Concrete bootstrap plan

1. Clone the harnesses and competitor repos under `references/repos/`.
2. Add thin LatentWire-side wrappers that emit JSONL predictions plus a `.meta.json` sidecar.
3. Run 5-10 example smoke tests first, not full tables.
4. Verify exact parser parity before comparing method rows.
5. Generate matched tables from telemetry only, with benchmark rows separated from method-ablation rows.
6. Promote one row at a time into the paper once the harness is stable.

## Good starter comparison ladder

- LatentWire target-alone control
- LatentMAS baseline
- LatentMAS text-MAS
- LatentMAS latent-MAS
- C2C projector baseline
- LatentWire strict route-selection ablations
- LatentWire shared-hub / tokenizer / verifier-stop ablations

## Story hygiene

The benchmark track should answer: "Can the harness reproduce fair comparisons?" It should not answer: "Is the method solved?"

The method story should answer: "Which communication rule survives matched comparisons?" Keep those sections separate until the final paper draft.
