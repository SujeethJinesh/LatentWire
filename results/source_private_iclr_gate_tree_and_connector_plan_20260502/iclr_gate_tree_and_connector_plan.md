# ICLR Gate Tree And Connector Plan

- date: `2026-05-02`
- readiness: COLM workshop plausible; ICLR full paper blocked.
- story: Fixed-byte source-private packets plus public benchmark gates and systems byte/exposure accounting.
- exact gap: A positive tokenwise learned/common-language connector and matched native systems rows are still missing.

## Technical Contributions

1. Fixed-byte source-private packet protocol with destructive controls.
2. Public-basis benchmark gates plus negative/falsification ladder.
3. Systems byte/exposure accounting against KV/cache transfer baselines.

## Next Exact Gate

Mac-local ARC n32 target-loss soft-prefix preflight using Qwen source features and a frozen target, followed by the OpenBookQA 3B train-only receiver gate if controls pass.

## Gate Tree

| Node | Status | Decision | Next Action |
|---|---|---|---|
| `fixed_byte_packet_protocol` | `promote_for_colm` | Keep as contribution, but do not call it solved latent reasoning. | Tie every method row to target-only, same-byte text, zero-source, wrong-row, and paired CI. |
| `public_basis_benchmark_gates` | `promote_for_colm` | Use as benchmark/evaluation contribution, not as final cross-family proof. | Add gate-tree figure and larger frozen slices when a connector clears validation. |
| `systems_boundary_accounting` | `promote_for_colm` | Promote as accounting; native GPU systems win remains blocked. | Fill native ingest rows for vLLM/SGLang/C2C/KVComm/KVCOMM/QJL/TurboQuant. |
| `tinyllama_mean_cache_connectors` | `cut` | Cut this family for ARC. | Do not spend more Mac cycles on mean-cache PCA/RFF/MLP variants. |
| `source_choice_scouts` | `cut_or_reprompt_only` | Cut current Llama/Phi source-choice senders; only revive with selected prompt/scoring. | If revived, select prompt/scoring on validation before frozen test. |
| `tokenwise_query_connector` | `alive_next_gate` | This is the next method gate; mean-cache proxies do not substitute for it. | Train frozen-endpoint 16-64 query connector with target loss and strict source-destroying controls. |
| `native_systems_rows` | `blocked_required` | Block native speed/HBM/goodput claims until measurements exist. | Run NVIDIA vLLM/SGLang/C2C/KVComm/KVCOMM/QJL/TurboQuant rows through the validator. |

## ICLR Needs

- Positive tokenwise query/soft-prefix or cache-fuser connector on a frozen larger slice.
- OpenBookQA 3B train-only receiver gate over packet-only with source-destroying controls.
- At least one strict true cross-family pair, not just same-family Qwen scaling.
- Seed repeats and paired uncertainty versus target-only, same-byte text, Qwen-substituted, wrong-row, and prompt/prefix controls.
- Direct comparison against C2C/KVComm/KVCOMM and KV quantization byte floors.
- Native NVIDIA rows for TTFT, TPOT, goodput, GPU memory, HBM/PCIe/NVLink traffic, accuracy, bytes, and source exposure.
- Interpretability telemetry: rate-distortion curve, effective rank or gate patterns, and source/help/harm decomposition.

## COLM Workshop Needs

- Scope as source-private packet protocol plus public benchmark positives.
- Use systems boundary table as accounting, not native speed.
- Include gate-tree figure showing promoted, cut, alive, and blocked branches.
- Cut HellaSwag receiver-improvement and TinyLlama hidden/query connector claims.
- Frame negative ladder as evidence of rigor, not as solved cross-model latent reasoning.

## Blockers For User Help

- NVIDIA access for target-loss tokenwise connector and native systems rows.
- Decision on COLM framing: workshop evidence package now, or hold for positive connector.
- Any preferred source/target model pair for the first expensive cross-family connector run.

## Tokenwise Connector Runbook

| Step | Name | Goal | Pass Rule |
|---:|---|---|---|
| 1 | Mac ARC smoke | Verify target-forward soft-prefix training code on 8-16 ARC rows without claiming evidence. | Training runs, controls are emitted, and no source answer/text/KV is transmitted. |
| 2 | ARC soft-prefix preflight | Train/select on ARC validation disagreement rows with frozen source and target endpoints. | Matched beats target-only, source-free, zero-source, row-shuffled, same-byte text, and Qwen-substituted controls. |
| 3 | OpenBookQA 3B receiver gate | Train-only receiver/query connector over the strongest OpenBookQA packet-only rows. | Packet-plus-receiver beats packet-only by >= +0.005 with positive paired CI95 low and source-destroying controls. |
| 4 | Frozen ARC/OpenBookQA test gate | Evaluate once on frozen larger slices after validation selection. | Mean delta >= 0.02 and CI95 low > 0 versus Qwen-substituted, packet-only, target-only, and same-byte text. |
| 5 | Cross-family falsification | Repeat with one true non-Qwen source-target pair. | Positive paired CI survives; same-family-only result is marked diagnostic. |
| 6 | Systems ingest | Fill native validator rows for LatentWire and all KV/cache baselines. | Validator allows native claims only after complete matched measurements. |

## Required Connector Controls

- target-only
- packet-only
- target-cache-only
- candidate-only
- target-derived packet
- row-shuffled source packet
- random same-rate packet
- label-permutation decoder
- candidate derangement
- same-byte visible text
- source-label-copy audit upper bound

## Subagent Synthesis

- `Pasteur`: Repo already has SVAMP target-loss soft-prefix and tokenwise query diagnostics, but ARC/OpenBookQA lack target-loss connector infrastructure. Decision: Implement a new isolated ARC/OpenBookQA soft-prefix preflight instead of editing core bridge modules.
- `Huygens`: Novelty survives only under per-example fixed-byte source-conditioned messaging, frozen/train-only receivers, and source-destroying controls. Decision: Use Prefix-Tuning, BLIP-2/Flamingo/Perceiver IO, C2C/KVComm, QJL, and TurboQuant as mandatory boundaries.

## Claim Boundary

- `unique`: Per-example source-conditioned rate-limited connector under source-destroying controls and explicit source-exposure accounting.
- `not_unique`: Generic soft prompts, static prefix tuning, and KV/cache fusion are prior art and must be baselines.
- `lateral_branches`: DiT/consistency-style iterative refinement and SAE universal feature spaces are inspirations, not current positive evidence.
