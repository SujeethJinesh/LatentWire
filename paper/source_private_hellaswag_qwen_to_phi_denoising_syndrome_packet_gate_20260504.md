# HellaSwag Qwen-To-Phi Denoising Syndrome Packet Gate

Date: 2026-05-04

## Readiness Status

Current paper readiness: COLM workshop remains plausible as a fixed-byte
packet/evaluation/systems-boundary paper; ICLR full is still blocked.

Current story: LatentWire has a strong source-private packet protocol and a
clean systems boundary. The live method question is whether a learned receiver
can use target-side evidence without destroying packet utility.

Exact blocking gap: this ridge-denoising syndrome branch does not beat the
fixed Qwen hybrid packet on the cached Phi receiver surface.

## Gate

Implemented and ran:

- `scripts/build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.py`
- `results/source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate_20260504_validation1024_2048/`

The gate treats Phi's four-choice scores as receiver-side information and fits
a tiny train-only ridge denoiser over candidate-level features. The source
packet is a discrete syndrome code derived from Qwen packet-policy predictions,
not source text, raw scores/logits, raw hidden vectors, or KV cache.

Split:

- fit: first `64` rows per cached `512`-row slice;
- select: next `64` rows per slice;
- eval: remaining `384` rows per slice, `768` total.

Selected source mode:

- `code8_hybrid_selected_margin`
- raw payload: `1B`
- framed record: `4B`
- ridge L2: `300.0`

## Result

The gate fails.

| Row | Accuracy | Delta vs fixed hybrid | CI95 low vs fixed hybrid |
|---|---:|---:|---:|
| target-or-hybrid oracle | `0.604167` | `+0.136719` | `+0.113281` |
| fixed Qwen hybrid | `0.467448` | reference | reference |
| denoising syndrome packet | `0.463542` | `-0.003906` | `-0.010417` |
| Qwen candidate-only | `0.455729` | `-0.011719` | `-0.023438` |
| Phi target-only | `0.263021` | `-0.204427` | `-0.250000` |
| zero-byte target ridge control | `0.252604` | `-0.214844` | `-0.263021` |
| target-derived-code control | `0.252604` | `-0.214844` | `-0.263021` |
| source-row shuffle control | `0.250000` | `-0.217448` | `-0.264323` |
| code-value permutation control | `0.248698` | `-0.218750` | `-0.268229` |
| random same-byte control | `0.218750` | `-0.248698` | `-0.295573` |
| candidate-roll code control | `0.190104` | `-0.277344` | `-0.330729` |
| label-permutation decoder control | `0.164062` | `-0.303385` | `-0.355469` |

Slice readout:

| Slice | Denoising | Fixed hybrid | Delta |
|---|---:|---:|---:|
| `1024:1536` | `0.486979` | `0.486979` | `0.000000` |
| `1536:2048` | `0.440104` | `0.447917` | `-0.007812` |

## Interpretation

The destructive controls collapse, which is good: the learned receiver does
depend on the source code semantics when it changes predictions. The problem is
that the selected code is too conservative and slightly harmful on heldout
rows: it creates `2` helps and `5` harms versus fixed hybrid.

Scientifically, this weakens a simple ridge-denoising syndrome packet. It does
not kill the broader rate-distortion/side-information framing because the
target-or-hybrid oracle remains `+13.67` points above fixed hybrid. The next
method must access that headroom without using eval labels or raw source state.

## Contribution Status

Promote:

- The Qwen-to-Phi receiver surface as a real cross-family headroom benchmark.
- The denoising syndrome gate as a negative control that prevents overclaiming
  a shallow receiver.
- The systems-compatible packet contract: `1B` raw / `4B` framed with no source
  text, KV, raw hidden vector, or raw score/logit vector.

Weaken:

- Simple ridge denoising over hand-coded source syndrome bits.
- Target-score-only repair models on this Phi surface.
- Any claim that the current receiver can exploit the `0.604167` oracle.

Still alive:

- A stronger denoising receiver that trains on more data, uses cross-fitted
  calibration, or learns a more structured source-code dictionary.
- A native systems path once NVIDIA rows are available.

## Lay Explanation

We tried to send Phi a tiny correction clue from Qwen. Phi used the clue plus
its own answer scores to decide whether to repair the Qwen hint. The clue was
meaningful enough that shuffled and permuted controls failed, but it did not
improve the final answer rate. It made a few good repairs and a few more bad
ones.

## Next Gate

Do not continue with hand-coded ridge denoising on this exact feature set. The
next highest-value method branch should be one of:

1. a cross-fitted learned source-code dictionary trained on a larger official
   HellaSwag train cache if we can afford Phi scoring on Mac;
2. a margin-constrained denoiser that is only allowed to override on rows with
   calibrated positive expected value;
3. native systems rows if NVIDIA access becomes available before a stronger
   method branch is ready.
