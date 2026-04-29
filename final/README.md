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
- codebook remapping across seeds preserves the result: on a `500`-example
  all-family deterministic gate, three distinct diagnostic codebooks pass at
  `2/4/8/16` bytes with exact ID and public candidate-label parity.

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

## Target-Model Decoder Ablation

The main result uses a deterministic protocol decoder to isolate source-side
communication. A Qwen3 target-decoder ablation now passes beyond tiny smoke:
core seed29 n64 on CPU reaches `42/64 = 0.656` matched packet accuracy versus
`16/64 = 0.250` target-only and `16/64 = 0.250` best control, with valid matched
predictions `1.000` and exact-ID parity true. The held-out seed30 n64 CPU row
also passes: `46/64 = 0.719` matched versus `16/64 = 0.250` target-only and
`17/64 = 0.266` best control. The attempted n160 MPS target decoder run is
backend-blocked by an Apple MPS matmul shape error. Treat the paired n64 rows as
ablations that reduce, but do not eliminate, the hand-coded decoder objection.

## Systems And Novelty Update

The systems artifact `results/source_private_systems_summary_20260428/` now
aggregates deterministic rate rows, model-produced packet rows, and
target-decoder rows. The main systems point is a rate frontier: 2-byte packets
reach `1.000` on deterministic core/held-out `500`-example surfaces while
matched-byte text remains at `0.250`; full hidden-log relay also reaches
`1.000` but costs `366.45-373.50` bytes, so the packet is
`183.2x-186.7x` smaller.

The novelty memo `references/481_systems_novelty_and_future_methods_refs.md`
positions the work against C2C, KVComm, activation communication, prompt
compression, text/tool-agent handoff, source coding with decoder side
information, quantization, JEPA, Q-Former, and diffusion-inspired successor
branches. The safest full-paper claim is source-private, extreme-rate
communication with decoder side information, not broad latent transfer.

## Learned Syndrome Packet Smoke

The learned-syndrome smoke in
`results/source_private_learned_syndrome_smoke_20260429/` is the current best
next method contribution candidate. It replaces the hand-designed diagnostic
packet with a tiny learned encoder that maps private source observations to a
1/2/4/8-byte binary syndrome decoded with target-side candidate latents.

Seed29 passes at 1/2/4 bytes: matched `0.820/0.949/0.992` versus target
`0.250`. Seed30 repeats the low-rate result at 1/2 bytes: matched
`0.797/0.902` versus target `0.250`. Higher budgets are not promoted because
some source-free controls rise above the tolerance. Treat this as a synthetic
learned-method smoke, not yet a headline claim.

The real-feature follow-up in
`results/source_private_tool_trace_learned_syndrome_20260429/` moves the same
idea onto hidden-repair tool-trace and candidate-text features. At the common
6-byte budget, seed pair `29 -> 30` reaches `0.945` matched accuracy versus
`0.250` target and `0.285` best no-source; seed pair `31 -> 32` reaches `0.918`
versus `0.250` target and `0.289` best no-source. This is now the strongest
learned-method candidate, but it still needs compression-native baselines before
becoming a headline claim.

## Latest-Model And MoE Status

The current evidence covers the final submitted model rows, including Qwen3,
Phi-3, and Qwen2.5-era source emitters. A post-package scout adds a matrix for
Qwen3.5 small models, Qwen3.6 MoE models, and non-Qwen cross-family falsification
rows under `results/source_private_latest_model_matrix_20260428/`.

The first latest-small row now passes: after upgrading the repo-local stack to
`transformers==5.7.0`, `Qwen/Qwen3.5-0.8B` passes the source-private packet gate
on CPU at `n=16`, `n=64`, and `n=160`, with n160 repeated on seeds `29` and
`31`. Matched packet accuracy is `1.000`, target/control floor is near `0.250`,
packet valid rate is `1.000`, and exact-ID parity holds. Apple MPS still fails
before generation in the model's
hybrid-attention matmul path, so this is CPU evidence only.

`Qwen/Qwen3.5-2B` also passes CPU n16, n64, and n160 rows with matched accuracy
`1.000`, target/control floor near `0.250`, and packet valid rate `1.000`,
adding a second n160-confirmed latest-small Qwen3.5 size. `Qwen/Qwen3.5-4B`
passes CPU n16 and n64 rows with matched `1.000`, controls `0.250`, and valid
packet rate `1.000`, but p50 CPU packet latency is high at about `27.2s` on the
n64 row.

The non-Qwen evidence is stronger now. `google/gemma-4-E2B-it` reaches
`160/160 = 1.000` on MPS with the strict trace-no-hint prompt on seeds `29` and
`31`, versus the `0.250` target/control floor and packet valid rate `1.000`.
It also clears a large local frozen slice at `n=500`: `500/500 = 1.000`
matched, `125/500 = 0.250` target-only, `126/500 = 0.252` best
source-destroying control, exact-ID parity true, and p50 packet latency about
`754 ms`.
The paired Gemma raw-log/no-trace ablation removes the private diagnostic trace
line and collapses to `40/160 = 0.250` with `0` valid packets.
`ibm-granite/granite-3.3-2b-instruct`
reaches `128/160 = 0.800` on CPU with the copied-helper prompt and
`101/160 = 0.631` under the strict trace-no-hint prompt on seeds `29/31`,
versus `0.250` target floor and `0.250-0.256` best controls. Its paired
raw-log/no-trace seed31 row collapses to `40/160 = 0.250` with `0` valid
packets. OLMo 2-0425-1B is a behavioral negative with zero valid packets, and
Granite MPS is backend-blocked.

MoE generalization is plausible because the source task is exact private-evidence
packet emission, not dense-model-specific latent transfer, but it is not yet a
paper claim. The safe addition is seed-stable latest-small evidence plus
non-Qwen strict-prompt evidence with prompt-contract sensitivity. Claim MoE
only after Qwen3.6 35B-A3B/FP8 pass off-machine under the same controls; see
`paper/docs/source_private_qwen36_moe_falsification_runbook_20260428.md`.

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
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_endpoint.py --help
./venv_arm64/bin/python scripts/build_source_private_tool_trace_baseline_pack.py --help
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py --help
./venv_arm64/bin/python scripts/build_source_private_latest_model_matrix.py --help
./venv_arm64/bin/python scripts/build_source_private_systems_summary.py --help
./venv_arm64/bin/python scripts/run_source_private_learned_syndrome_smoke.py --help
./venv_arm64/bin/python scripts/run_source_private_tool_trace_learned_syndrome.py --help
./venv_arm64/bin/python scripts/run_source_private_tool_trace_compression_baselines.py --help
```

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_hidden_repair_packet_smoke.py \
  tests/test_run_source_private_hidden_repair_packet_llm.py \
  tests/test_run_source_private_hidden_repair_packet_endpoint.py \
  tests/test_build_source_private_tool_trace_baseline_pack.py \
  tests/test_build_source_private_tool_trace_figures.py \
  tests/test_run_source_private_tool_trace_target_decoder_smoke.py \
  tests/test_build_source_private_systems_summary.py \
  tests/test_run_source_private_learned_syndrome_smoke.py \
  tests/test_run_source_private_tool_trace_learned_syndrome.py \
  tests/test_run_source_private_tool_trace_compression_baselines.py -q
```

## Latest Method Pivot

The live method is now the 6-byte scalar-quantized learned source projection.
On the `768/512` frozen tool-trace compression gate it reaches `0.979` accuracy
versus learned sign syndrome `0.953`, target-only `0.250`, raw source sign
sketch `0.307`, scalar shuffled source `0.166`, and scalar answer-masked source
`0.293`. The next blocker is not a local artifact issue; it is the research
gate for 5-seed repeats, held-out-family splits, codebook remap, and
candidate-side masking.

## Final Status

The reproducibility snapshot is current through the scalar-packet compression
gate. The paper is not yet final-submission complete: the remaining blocker is
research evidence, specifically scalar-packet repeats and stricter
generalization/leakage gates before venue upload.
