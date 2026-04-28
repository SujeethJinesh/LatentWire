# Final Reproducibility Folder

This folder collects the final paper package, upload artifacts, code snapshots,
tests, result artifacts, references, and handoff notes for the source-private
tool-trace communication paper.

## Paper Story

The paper asks a narrow but important question:

Can one model or agent communicate private evidence to another model without
sending the full private trace, and can we prove the gain is real communication
rather than target priors, formatting artifacts, or answer leakage?

Our answer is a scoped positive method: a source agent observes private
tool-trace diagnostics and emits a rate-capped diagnostic packet. A target-side
decoder has the public prompt and a candidate pool. The packet selects the
candidate consistent with the private evidence.

The supported claim is not broad latent transfer or universal agent
communication. The supported claim is explicit source-private diagnostic-code
communication with strict source-destroying controls and a large frozen
decision surface.

## Why This Matters

Modern AI systems are increasingly multi-agent and tool-using. One component may
see private logs, tool traces, retrieval evidence, tests, memory, or telemetry
that another component does not see. Passing the full trace is expensive,
latency-heavy, privacy-sensitive, and often unnecessary.

This paper shows a controlled way to study that handoff:

- the source has private evidence;
- the target has public task context and decoder side information;
- the source sends a tiny message;
- controls destroy or scramble source evidence;
- gains must disappear under those controls.

That makes the method useful as a benchmark and protocol for separating real
source-private communication from target-only cache/prior effects.

## Who Should Care

- Multi-agent LLM system builders who need compact handoffs between agents.
- Tool-use and code-repair researchers who want to transmit test/log evidence
  without replaying full traces.
- Model-routing and serving teams interested in byte/latency-efficient
  side-channel protocols.
- Interpretability and evaluation researchers who want source-destroying
  controls for communication claims.
- Reviewers of agent communication papers who need a sharper standard than
  "agent A helped agent B."

## What Worked

On the final tool-trace surfaces:

- target-only and source-destroying controls stay near the constructional
  candidate prior of about `25%`;
- matched source packets recover the private diagnostic signal;
- deterministic packets reach oracle-level behavior;
- Qwen3 and Phi-3 source packet emitters work strongly on 500-example frozen
  slices;
- raw-log/no-trace ablations collapse back to target-only, supporting that the
  explicit private diagnostic line is the transferred signal;
- structured text relay only explains the result at larger budgets, while the
  diagnostic packet works at the far-left low-rate regime.

Representative headline results:

- core seed29: Qwen3 `404/500 = 0.808`, Phi-3 `1.000`, target `0.250`, best
  control about `0.252`;
- core seed31: Qwen3 `0.808`, Phi-3 `1.000`, target `0.250`;
- heldout seed30/32: Qwen3 about `0.922-0.924`, Phi-3 `1.000`, target `0.250`;
- strict-small Qwen2.5-0.5B: `150/160 = 0.938` versus `40/160 = 0.250`.

## What This Is Not

This is not a broad state-of-the-art claim over all cross-model communication.
It is not learned latent transfer, arbitrary KV/cache transport, or general raw
log repair. The contribution is a reproducible, interpretable, rate-capped
source-private communication protocol with strong controls.

## Latest-Model And MoE Status

The current evidence covers the final submitted model rows, including Qwen3,
Phi-3, and Qwen2.5-era source emitters. A post-package scout adds a matrix for
Qwen3.5 small models, Qwen3.6 MoE models, and non-Qwen cross-family falsification
rows under `results/source_private_latest_model_matrix_20260428/`.

The first latest-small row now passes: after upgrading the repo-local stack to
`transformers==5.7.0`, `Qwen/Qwen3.5-0.8B` passes the source-private packet gate
on CPU at `n=16` and `n=64` with matched packet accuracy `1.000`, target/control
floor `0.250`, packet valid rate `1.000`, and exact-ID parity true. Apple MPS
still fails before generation in the model's hybrid-attention matmul path, so
this is CPU evidence only.

MoE generalization is plausible because the source task is exact private-evidence
packet emission, not dense-model-specific latent transfer, but it is not yet a
paper claim. Claim latest-model breadth only after Qwen3.5 widens to n160 with a
seed repeat and at least one non-Qwen row passes; claim MoE only after Qwen3.6
35B-A3B/FP8 pass off-machine under the same controls.

## Directory Map

- `upload/`: files intended for external upload and checksum sidecars.
- `paper/iclr2026/`: final manuscript source, bibliography, figures, and PDF.
- `paper/docs/`: paper memos, final review, upload audit, and handoff notes.
- `code/scripts/`: source-private experiment and build scripts.
- `code/tests/`: focused tests for the final method family.
- `results/`: relevant result directories and manifests.
- `references/`: local citation and literature memo files used in framing.
- `repo/`: top-level repo guidance and environment files.
- `MANIFEST.sha256`: checksum manifest for this `final/` folder.

## Upload Files

For double-blind submission, use:

- `upload/source_private_tool_trace.pdf`
- `upload/source_private_tool_trace_iclr_source_20260428.zip`
- `upload/source_private_tool_trace_artifacts_anonymous_20260428.zip`
- `upload/source_private_tool_trace_anonymous_upload_checksums_20260428.sha256`

Convenience transfer bundle:

- `upload/source_private_tool_trace_anonymous_submission_bundle_20260428.zip`

The portal may require the PDF, source zip, and artifact zip as separate files.
The transfer bundle is for moving them together.

## Verification

From the repository root:

```bash
shasum -a 256 -c paper/source_private_tool_trace_anonymous_upload_checksums_20260428.sha256
unzip -t paper/artifacts/source_private_tool_trace_anonymous_submission_bundle_20260428.zip
```

From inside `final/`:

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Reproduction Entry Points

Run from the repository root with `./venv_arm64/bin/python`.

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py --help
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --help
./venv_arm64/bin/python scripts/build_source_private_tool_trace_baseline_pack.py --help
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py --help
./venv_arm64/bin/python scripts/build_source_private_latest_model_matrix.py --help
```

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_hidden_repair_packet_smoke.py \
  tests/test_run_source_private_hidden_repair_packet_llm.py \
  tests/test_build_source_private_tool_trace_baseline_pack.py \
  tests/test_build_source_private_tool_trace_figures.py \
  tests/test_run_source_private_tool_trace_target_decoder_smoke.py -q
```

## Final Status

Local preparation is complete. The remaining blocker is external only: upload
through the venue-approved anonymous route, then record submission/artifact IDs
and portal warnings in the submission confirmation memo.
