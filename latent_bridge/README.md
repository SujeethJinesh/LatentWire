# RotAlign-KV

**Cross-model KV-cache transfer via rotational alignment and Lloyd-Max quantization.**

See [`method.md`](method.md) for the formal writeup, math, and experimental protocol.

## Core idea in one paragraph

KV caches from different LLMs live in anisotropic, outlier-heavy, model-specific
coordinate systems, which is why prior cross-model methods (C2C, KVComm) need
heavy learned MLP fusers. We apply a fixed random (or Hadamard) rotation to
both models' KV spaces, optionally ZCA-whitened first, which Gaussianizes them.
In Gaussianized coordinates, cross-model alignment collapses to a closed-form
linear problem — orthogonal Procrustes, ridge, CCA, or reduced-rank regression
depending on the shape regime. Since the coordinates are now Gaussian, a scalar
Lloyd-Max quantizer is near-optimal, giving us compression to 3–4 bits
essentially for free. The full pipeline has a closed-form fit, a single
trainable fusion gate, and no deep networks anywhere.

```
  source KV ─┐
             ▼
    [ optional ZCA whitening  ]   (anisotropy correction)
             │
             ▼
    [ rotation (orthogonal or  ]   (Gaussianize, training-free)
    [ Hadamard: O(d log d))    ]
             │
             ▼
    [ linear alignment:        ]   (closed-form)
    [ Procrustes / ridge / CCA ]
    [ / reduced-rank / rand-   ]
    [ SVD Procrustes for 70B   ]
             │
             ▼
    [ Lloyd-Max b-bit quantize ]   (near-optimal for Gaussian source)
             │
             ▼
    [ rotate back to native    ]
             │
             ▼
    [ gated fusion into target ]   (learnable scalar gate per layer)
```

## Repository layout

```
rotalign_kv/
├── method.md                     Formal writeup: math, protocol, contributions
├── README.md                     This file
├── requirements.txt
├── rotalign/                     Core library
│   ├── __init__.py               Public API
│   ├── rotation.py               Haar rotation, Hadamard, ZCA whitening
│   ├── procrustes.py             5 closed-form alignment solvers
│   ├── quantize.py               Lloyd-Max scalar quantizer (searchsorted-optimized)
│   └── translator.py             RotAlignKVTranslator composing all stages
└── scripts/
    ├── demo.py                   Self-contained sanity check (no model downloads)
    ├── calibrate.py              Fit translator on real HF models
    └── evaluate.py               Compare against baselines on MCQ and generation tasks
```

## Component study

The method is designed as a swappable pipeline. Each stage has multiple options,
and the ICLR paper reports which combination wins:

| Stage | Options | Config flag |
|---|---|---|
| Rotation | `identity`, `orthogonal`, `hadamard` | `rotation_kind` |
| Whitening | off, ZCA on source | `use_whitening` |
| Alignment | `identity`, `procrustes`, `ridge`, `cca`, `reduced_rank`, `procrustes_rand` | `alignment_method` |
| Quantization | 2 / 3 / 4 / 6 / 8 bits (or off) | `quant_bits` |
| Layer pairing | linear interp, CKA-ranked, explicit list | `layer_pairing` |
| Layer selection | all layers, top-k, ratio-based | `layer_selection_*` |
| Fusion gate | fixed 0.5, learned, line-searched | (trained on gate_K/V) |
| Protocol | translated-only, fused, text+KV hybrid | `evaluate.py` method modes |

Everything is exposed via `TranslatorConfig` and via CLI flags on
`scripts/calibrate.py`, so a factorial ablation sweep is just a bash loop
over flag combinations.

## Quickstart

### 1. Install

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the sanity check

Verifies the full pipeline mechanically on synthetic data with no model downloads.
Runs in under a minute on CPU:

```bash
python scripts/demo.py
```

This checks: exact orthogonality of both rotation variants, Gaussianization
under rotation (kurtosis drop), ZCA whitening (variance equalization across
dimensions), all five alignment solvers on planted structure, Lloyd-Max
distortion tracking the theoretical rate-distortion bound, and the full
translator on mismatched-shape synthetic KVs.

### 3. Calibrate on real models

Create a text file with one prompt per line (any reasonable instruction-tuning
prompts will do; 500–1000 is plenty), then fit:

```bash
python scripts/calibrate.py \
    --source-model Qwen/Qwen2.5-0.5B-Instruct \
    --target-model Qwen/Qwen3-0.6B \
    --calibration-file data/calibration.txt \
    --output checkpoints/qwen25_to_qwen3.pt \
    --bits 4 \
    --rotation orthogonal \
    --alignment auto \
    --verbose
```

**New flags since the first version:**

- `--rotation {identity, orthogonal, hadamard, dct}` — choose rotation variant.
  `identity` is the no-rotation ablation. Hadamard is O(d log d) and scales
  to very large head dims; `dct` is the Fourier-family dense-mixing control.
- `--whitening` — apply ZCA whitening of source coordinates before alignment.
  Helps when the two models have very different coordinate scales.
- `--alignment {auto, identity, procrustes, procrustes_rand, ridge, cca, reduced_rank}` —
  pick the alignment solver. `auto` uses Procrustes if dimensions match and
  ridge otherwise. Use `cca` when you suspect the cross-model map is
  primarily low-rank and partially diagonal; use `reduced_rank` with
  `--alignment-rank 64` when you want explicit compression.
- `--alignment-rank N` — rank for CCA and reduced-rank regression.
- `--layer-pairing {interp,cka,reverse,shifted,random}` — interpolation,
  SemAlign-style CKA pairing, or negative-control layer maps.
- `--layer-selection-topk K` / `--layer-selection-ratio R` — selective
  transmission ablations inspired by KVComm-style sparsity.

### 4. Evaluate against baselines

```bash
python scripts/evaluate.py \
    --translator checkpoints/qwen25_to_qwen3.pt \
    --source-model Qwen/Qwen2.5-0.5B-Instruct \
    --target-model Qwen/Qwen3-0.6B \
    --eval-file data/gsm8k_eval_70.jsonl \
    --task-type generation \
    --methods target t2t rotalign rotalign_translated rotalign_text_kv \
    --gate-mode search \
    --gate-search-file data/gsm8k_gate_search_30.jsonl \
    --gate-search-limit 30 \
    --gate-values 0.15 0.25 0.30
```

Supports MCQ and exact-match generation tasks. The default `rotalign` mode is
the fused protocol that actually exercises the target-side fusion gate.
Additional method modes expose translated-only and text+KV hybrid ablations.
Pass `--no-quantize` to ablate the Lloyd-Max round-trip.

For the current control pair, prefer held-out gate search over eval-set gate
Sweeps. The earlier `0.06` pilot on `GSM8K-100` was directionally useful, but
it selected the best gate on the same slice it reported on. The next cycle
should use `data/gsm8k_gate_search_30.jsonl` for gate selection and
`data/gsm8k_eval_70.jsonl` for the reported score.

## Focused control-suite runner

Use the focused control suite for the next fairness-corrected control cycle:

```bash
python scripts/run_control_suite.py \
    --calibration-file data/calibration.txt \
    --eval-file data/gsm8k_eval_70.jsonl \
    --gate-search-file data/gsm8k_gate_search_30.jsonl \
    --results-dir results/control_suite_rerun \
    --checkpoint-dir checkpoints/control_suite_rerun
```

The default suite now does three things the earlier pilot did not:
- runs matched `text-to-text` baselines under `plain`, `brief_analysis`, and `cot`
- reports translated-only and text+KV protocol comparisons alongside fused KV
- uses a held-out gate-selection split when `--gate-search-file` is provided

## Running ablations

The factorial sweep for the ICLR paper is a bash loop. Example for the
rate-distortion curve:

```bash
for bits in 2 3 4 6 8; do
    python scripts/calibrate.py \
        --source-model Qwen/Qwen2.5-0.5B-Instruct \
        --target-model Qwen/Qwen3-0.6B \
        --calibration-file data/calibration.txt \
        --output checkpoints/qwen_bits${bits}.pt \
        --bits $bits
    python scripts/evaluate.py \
        --translator checkpoints/qwen_bits${bits}.pt \
        --source-model Qwen/Qwen2.5-0.5B-Instruct \
        --target-model Qwen/Qwen3-0.6B \
        --eval-file data/mcq.jsonl | tee results/bits${bits}.txt
done
```

For the rotation ablation, sweep `--rotation` across `identity`, `orthogonal`,
`hadamard`, and `dct`. For the alignment ablation, sweep `--alignment`. For the
whitening ablation, toggle `--whitening`. For pairing / sparsity / protocol
ablations, sweep `--layer-pairing`, `--layer-selection-*`, `--gate-*`, and
the `rotalign_*` evaluate modes. Use `--source-kv-control` for random/zero/
shuffled-source negative controls, `--translated-kv-control` for stricter
post-translation target-space controls, and `--quantization-control
matched_noise` to separate true discretization from noise smoothing. In the
control suite, `target_attenuation_brief` is the zero-byte target-cache
attenuation baseline; source-communication claims require real translated KV to
beat that baseline on paired examples. Use `--no-quantize` as the
full-precision anchor before comparing 4-bit and lower-bit runs.

## For the paper: what to run in what order

**Immediate pilot matrix (M1-friendly, 64 GB unified memory):**

1. `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
2. `Qwen/Qwen2.5-0.5B-Instruct -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
3. `Qwen/Qwen2.5-0.5B-Instruct -> google/gemma-4-E2B-it`
4. `Qwen/Qwen3-0.6B -> google/gemma-4-E2B-it` if you want a second
   cross-tokenizer stress test
5. Manual follow-up only, not unattended default: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-0.8B`
6. Later stretch: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3.5-4B`

This ordering keeps the first pass focused on the control pair, a reasoning-tuned
receiver, and one cross-tokenizer stress test. Qwen3.5 is a follow-up target, not
the default unattended run.

**Workshop (COLM, June 23 deadline):** minimum viable experiment set.

1. Re-run the control pair on a held-out `30/70` GSM8K split with gate search
   on the `30`-example dev slice and reporting only the `70`-example eval score.
2. Run a fairness baseline sweep: `text-to-text` under `plain`,
   `brief_analysis`, and `cot`, plus matched fused / translated-only / text+KV
   latent runs under the same prompting.
3. Add one knowledge task before widening the model matrix: `MMLU-Redux` or
   `ARC-C`, using the same best control-pair config.
4. Only after that, add one cross-family stress test:
   `Qwen/Qwen2.5-0.5B-Instruct -> google/gemma-4-E2B-it`
5. Rate-distortion sweep: bits ∈ {2, 3, 4, 8}
6. 4 core ablations: no-rotation, identity-W, interp-vs-CKA, full-precision-vs-4-bit
7. Follow-on structural studies: head-grouped / per-head alignment, steering-vector baseline
8. Report accuracy, bytes transmitted, TTFT, decode throughput, and end-to-end latency
9. ~10–20 GPU-hours equivalent, depending on calibration size and cache reuse

**Full paper (ICLR 2027, late September):** component study + phased matrix.

1. Factorial sweep on one control pair and one reasoning task
2. Freeze the winning configuration
3. Run the full benchmark on the M1-friendly matrix above
4. Add larger or harder follow-ons: `Qwen/Qwen3.5-4B`, Llama controls, and explicit cross-tokenizer stress tests
5. Add direct C2C and KVComm baseline runs on the strongest small-pair settings
6. Include selective layer transmission and systems metrics as first-class results
7. ~150–200 GPU-hours if the full benchmark is expanded to the larger matrix

## Known limitations

- **Cross-tokenizer calibration** assumes approximate token-wise pairing.
  Best results come from same-tokenizer pairs (Qwen2.5 ↔ Qwen3, Llama-3 ↔ Llama-3.1).
  Treat Qwen → Gemma and similar pairs as stress tests until anchor-style or
  Gromov-Wasserstein alignment is added.
- **Headline control results must use held-out gate selection.**
  The current best local pilot (`0.06` on a 100-example GSM8K slice) came from
  picking the best gate on the same slice it was reported on. Treat it as
  directional until the `30/70` held-out rerun is done.
- **Scalar quantization is suboptimal** for correlated dimensions. If residual
  correlation survives rotation, vector quantization (full TurboQuant) would
  outperform.
- **Calibration fits in CPU memory.** For very large models with ~60 layers
  and long calibration sequences, switch to streaming / randomized Procrustes.

## Key references

- **TurboQuant** (rotation + scalar quantization front-end): Zandieh, Daliri,
  Hadian, Mirrokni. ICLR 2026. arXiv:2504.19874.
- **QuIP#** (Hadamard rotation trick): Tseng et al. NeurIPS 2024.
- **Cache-to-Cache (C2C)** (learned KV fuser baseline): Fu et al. ICLR 2026. arXiv:2510.03215.
- **KVComm** (selective layer sharing baseline): Shi et al. ICLR 2026. arXiv:2510.03346.
- **SemAlign** (CKA layer pairing): Gu et al. arXiv:2510.24208.
- **Latent Space K-V Cache Alignment** (shared-latent-space variant): Dery et al. arXiv:2601.06123.
- **Relative representations** (zero-shot stitching foundation): Moschella et al. ICLR 2023. arXiv:2209.15430.
- **Platonic Representation Hypothesis** (theoretical motivation): Huh, Cheung, Wang, Isola 2024. arXiv:2405.07987.
