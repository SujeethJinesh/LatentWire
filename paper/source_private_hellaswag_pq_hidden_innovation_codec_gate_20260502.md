# HellaSwag PQ Hidden Innovation Codec Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible around the fixed-byte
  source-private packet and systems-rate evidence; ICLR remains blocked by a
  positive learned receiver/common-language method.
- Current story: HellaSwag has a strong compact TinyLlama packet and real
  TinyLlama/Qwen complementarity, but score-code, receiver-selector,
  sparse-query, crosscoder, and now product-quantized hidden-code branches do
  not improve over packet-only.
- Exact remaining blocker: a true connector or a materially different
  benchmark/source-information surface must beat packet-only with paired
  uncertainty and destructive controls.

## Lay Explanation

This experiment asks whether TinyLlama can send more than just its answer
choice without sending a hidden vector. It compresses a hidden-state fingerprint
into several tiny product-quantized subcodes, packs those subcodes plus the
answer id into one byte, and lets Qwen use that byte with its own scores.

## Artifact

`results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/hellaswag_pq_hidden_innovation_codec_gate.json`

Supporting files:

- `results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/hellaswag_pq_hidden_innovation_codec_gate.md`
- `results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/manifest.json`

## Method

- Calibration: official HellaSwag train cache rows, `1487` retained rows after
  prior duplicate/out-of-bag filtering, split `1115/372`.
- Evaluation: frozen validation slice `1024:2048`, `1024` rows, reusing the
  cached TinyLlama hidden-state file from the earlier hidden-code scout.
- Packet contract: `1B` raw / `4B` framed.
- Source code: PCA over TinyLlama packet-candidate hidden residual features,
  optional train-independent orthogonal rotation, product quantization over
  factorized subspaces, then `subcode * 4 + source_candidate`.
- Decoder: candidate-wise ridge decoder with Qwen score side information.
- Controls: row-shuffled PQ code, hidden-feature shuffle before encoding,
  codebook permutation mismatch, random same-byte code, candidate-only code,
  zero source code, label-permutation decoder, and packet-only.

## Result

The gate fails.

| Row | Accuracy | Packet-only | Delta | CI95 low |
|---|---:|---:|---:|---:|
| predeclared default | `0.497070` | `0.501953` | `-0.004883` | `-0.017578` |
| best diagnostic scout | `0.508789` | `0.501953` | `+0.006836` | `0.000000` |

Default details:

- encoder: `pq_pca16_m4_k2_identity`;
- codebook size: `64`;
- unique eval codes: `64`;
- ridge: `0.01`;
- below prior hidden-code slice scout `0.511719` by `-0.014649`;
- block-stability gate: `false`;
- control-separation gate: `false`.

The default is positive in only one of five contiguous blocks. The best scout
does not clear the predeclared `+0.010` scout bar, has CI95 low exactly `0`,
and remains below the prior anchor-relative hidden-code near-miss.

## Controls

Destructive controls show the code path is sensitive, but not useful:

| Control | Accuracy | Delta vs packet-only |
|---|---:|---:|
| row-shuffle PQ code | `0.284180` | `-0.217773` |
| hidden-feature shuffle before encoding | `0.309570` | `-0.192383` |
| codebook permutation mismatch | `0.292969` | `-0.208984` |
| random same-byte code | `0.283203` | `-0.218750` |
| candidate-only code | `0.501953` | `0.000000` |
| zero source code | `0.464844` | `-0.037109` |
| label-permutation decoder | `0.260742` | `-0.241211` |

## Systems Readout

- packet remains `1B` raw / `4B` framed;
- logical slice payload is `1024B` raw / `4096B` framed;
- no source text, source KV, raw hidden vectors, or raw score vectors are
  transmitted;
- cached Mac-local total wall time was `87.80s`;
- native GPU systems claims remain disabled.

## Interpretation

This branch does not promote. It was the right remaining Mac-local
falsification after source-score codes failed because it changes the source
information to hidden-state features and uses a product-quantized rate point
inspired by vector-quantization/KV-quantization work. The result says that
factorized TinyLlama hidden-code packets still do not add stable task utility
beyond the compact candidate packet on this HellaSwag slice.

## Contribution Status

Promoted:

1. A stricter negative hidden-code ablation that tests product quantization,
   not only PCA/k-means, anchor-relative coordinates, or linear crosscoders.
2. A clearer branch-kill decision for Mac-local HellaSwag hidden-code scouts.
3. A systems-accounting row preserving the one-byte source-private packet
   boundary.

Still blocked:

1. Positive receiver improvement over packet-only.
2. Benchmark diversity beyond the current packet-saturated HellaSwag surface.
3. Native NVIDIA/vLLM/SGLang systems measurements against C2C/KVComm and
   KV-quantization baselines.

## Decision

Stop widening Mac-local HellaSwag hidden-code/codebook variants. For ICLR, cut
the claim that HellaSwag currently has receiver-improvement evidence and keep
it as fixed-byte systems evidence, complementarity/headroom, and negative
ablation. The next exact positive-method gate should be either a true learned
query/cache connector on NVIDIA or a second benchmark where the compact
candidate id does not already saturate the source signal.
