# E2 Preflight Scope Decision

Created: 2026-05-26T11:44:35Z

## Context

The revised Stage 1 order authorizes E2 cross-prompt replication after E3,
with MATH-500 plus GPQA-Diamond if feasible and a documented MATH-only fallback
if GPQA is not feasible within the run constraints.

## Finding

Preflight in the repo-local `.venv_gpu` environment found that `datasets` was
missing. I installed `datasets==4.8.4` into the repo-local venv and added the
same exact dependency to `requirements.txt` and `release/pyproject.toml`.

MATH-500 loaded successfully from `HuggingFaceH4/MATH-500`. GPQA-Diamond via
`Idavidrein/gpqa` failed because the Hugging Face dataset is gated in this
environment:

> Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub.

## Decision

Run E2 with the explicit `--math-only` fallback unless authenticated GPQA access
is supplied before E2 starts. This is a routine scope reduction under the
authorized Stage 1 rules, not a scientific reframing. The paper limitation
should state that the cross-prompt replication uses MATH-500 in this autonomous
run, with GPQA-Diamond deferred because access was gated.

## Guardrail

The E2 runner was patched to capture and write progress one prompt at a time so
the 15-hour cap can stop cleanly between prompts instead of overrunning inside a
single model.
