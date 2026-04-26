# Cross-Family Interface Proxy Gate

- date: `2026-04-26`
- scale-up rung: micro smoke / strict-surface scout
- status: `proxy_surface_fails`
- code commit entering cycle: `b5ec164e`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target models scouted: `Qwen/Qwen3-0.6B`, `microsoft/Phi-3-mini-4k-instruct`,
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `facebook/opt-350m`
- live branch entering cycle: quotient/GPA sparse dictionary plus
  sequence-aligned byte sidecar, using byte-span module replacement as the
  closest real-model proxy

## Readiness Snapshot

Current ICLR readiness: not ready. The paper still lacks a stable,
source-derived positive method that beats target-alone, text/token relay, and
C2C or gives a clear systems tradeoff under source-destroying controls.

Current story: C2C and target repair expose headroom, raw dynalign is
seed-fragile, process repair selects target-only candidates, and the only live
technical clue is that a shared quotient/GPA sparse basis benefits from
byte/sequence-aligned sidecars under tokenizer-like corruption.

Exact blocker: no deployable real-model sidecar/transport row has survived a
strict exact-ID gate with source controls.

## What Ran

1. Real tokenizer/interface scout on `data/gsm8k_gate_search_30.jsonl`.
2. Toy sequence-aligned sidecar seed repeat.
3. Cross-family byte-span module-replace harness repairs for decoder-layer
   targets, packed `qkv_proj`, and OPT projection width mismatch.
4. Qwen2.5 -> OPT-350m byte-span module-replace calibration and matched GSM30
   generation.

## Tokenizer Scout

| Pair | Shared Decoded | Boundary F1 | Frag Delta | Remap Coverage |
|---|---:|---:|---:|---:|
| Qwen2.5 -> Qwen3 | 1.0000 | 1.0000 | 0.0000 | 0.0550 |
| Qwen2.5 -> Phi-3 | 0.7972 | 0.9347 | 0.0225 | 0.0706 |
| Qwen1.5 -> Llama1 | 0.9256 | 0.9843 | -0.0074 | 0.0550 |
| Qwen2.5 -> OPT-350m | 0.9047 | 0.9434 | -0.0052 | 0.0550 |

Interpretation: Phi-3 is the strongest tokenizer-mismatch surface, but it
exposes packed-QKV/GQA compatibility issues. OPT is runnable after harness
repairs, but its GSM30 target surface is too weak to be a useful decision
surface.

## Toy Seed Repeat

Seed `1` preserves the low-shot clue:

| Shot | Best Shared-Basis Sequence Sidecar MSE |
|---:|---:|
| 1 | 0.0362 |
| 2 | 0.0361 |
| 4 | 0.0361 |
| 8 | 0.0361 |

The sequence-aligned sidecar remains the best shared-basis branch, but direct
held-out few-shot byte-span remap is better once paired data reaches `4+`
shots/class. This keeps the sidecar branch alive only as an interface component,
not as paper evidence.

## Harness Fixes

- `latent_bridge/calibrate.py`
  - added support for decoder-layer containers such as OPT
  - added packed `qkv_proj` query extraction
  - projected OPT output embedding rows through `decoder.project_out` when
    teacher-output width differed from target KV width
- `latent_bridge/evaluate.py`
  - added decoder-layer containers
  - added packed `qkv_proj` query extraction
- tests added for decoder containers, packed QKV, and projected output rows

Focused tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py tests/test_calibrate_and_ablation.py -q
```

Result: `204 passed`.

## OPT Proxy Calibration

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model facebook/opt-350m \
  --calibration-file .debug/calibration_64.txt \
  --output .debug/qwen25_phi3_bytespan_interface_20260426/qwen25_to_opt350m_bytespan_r4_cal64.pt \
  --device mps \
  --dtype float32 \
  --quantization-correction bridge_ridge_qk_bytespan_module_replace \
  --quantization-correction-rank 4 \
  --source-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false
```

Calibration readout:

- byte-aligned token pairs: `2702`
- changed vs spanalign prompts: `32/64`
- K cosine: `0.892`
- V cosine: `0.612`
- K relative Frobenius error: `0.424`
- V relative Frobenius error: `0.797`

## OPT Matched GSM30 Gate

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/qwen25_phi3_bytespan_interface_20260426/qwen25_to_opt350m_bytespan_r4_cal64.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model facebook/opt-350m \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --task-type generation \
  --methods target source t2t rotalign \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --prediction-output .debug/qwen25_phi3_bytespan_interface_20260426/outputs/qwen25_to_opt350m_bytespan_gsm30_matched.jsonl
```

Result:

| Method | Correct | Mean Latency Sec | TTFT Sec | Bytes / Example |
|---|---:|---:|---:|---:|
| target-alone | 0/30 | 3.3365 | 0.2039 | - |
| source-alone | 0/30 | 4.2258 | 0.3184 | - |
| text-to-text | 3/30 | 6.7272 | 3.6428 | prompt text |
| byte-span rotalign proxy | 0/30 | 4.0911 | 1.1669 | 525562.6 |

Decision: kill OPT-350m as a decision surface for this branch. It is too weak
on GSM30 and the byte-span proxy provides no matched lift while spending far
more bytes than the text relay.

## Branch Decision

- killed: OPT-350m cross-family proxy surface
- weakened: byte-span module replace as a standalone real method
- kept alive: quotient/GPA sparse dictionary plus sequence-aligned byte sidecar
  as an interface component, because the toy repeat remains stable
- promoted next: Phi-3/TinyLlama-compatible harness work only if it directly
  enables a stronger target surface; otherwise move to same-family Qwen3 exact
  gate or source-surface discovery with C2C residual targets

## Next Exact Gate

Run a strict small gate on a target surface with nonzero target/text headroom:

1. Preferred: Qwen2.5 -> Phi-3 or TinyLlama after GQA/packed-QKV evaluation
   support, only if baseline target/text accuracy is nonzero on the exact-ID
   slice.
2. Fallback: Qwen2.5 -> Qwen3 exact-ID SVAMP32/GSM32 with the existing
   strongest output-aware transport plus an explicit sidecar component and the
   full source-destroying control suite.

Promotion rule remains:

- matched `>=14/32`
- target-self `3/3`
- clean source-necessary `>=2/6`
- numeric coverage `>=31/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, target-only, and slots-only
  controls have clean union `0/6`

