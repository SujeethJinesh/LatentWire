# 362 LatentMAS Competitor Bootstrap References

Date: 2026-04-21

Scope: bootstrap LatentMAS as a direct competitor/reference lane for LatentWire. This memo does not edit paper docs and does not stage cloned repository contents.

## Local Repository State

- Local clone: `references/repos/LatentMAS`
- Remote: `https://github.com/Gen-Verse/LatentMAS.git`
- Current local commit: `b9b2095 Update README`
- Clone status: present and clean at inspection time.
- Tracking convention: this repository ignores `references/repos/`, so cloned competitor contents should remain unstaged unless the project convention changes.

## Primary Sources

- LatentMAS paper: https://arxiv.org/abs/2511.20639
- LatentMAS repository: https://github.com/Gen-Verse/LatentMAS
- Hugging Face paper page: https://huggingface.co/papers/2511.20639
- Local README anchor: `references/repos/LatentMAS/README.md`
- Local entrypoint: `references/repos/LatentMAS/run.py`
- Local method implementations:
  - `references/repos/LatentMAS/methods/baseline.py`
  - `references/repos/LatentMAS/methods/text_mas.py`
  - `references/repos/LatentMAS/methods/latent_mas.py`

LatentMAS claims a training-free latent multi-agent system that replaces text inter-agent messages with latent working memory. The paper/repo position it against single-agent and text-MAS controls, reporting accuracy gains with substantially lower output-token and wall-clock costs. This is directly relevant as a same-model latent-communication competitor, but it is not the same problem as LatentWire cross-model communication: LatentMAS primarily transports latent working memory inside one model family/backbone, while LatentWire must handle cross-model orientation, tokenizer, cache, and route-selection mismatch.

## Harness Findings

- Native methods: `baseline`, `text_mas`, `latent_mas`.
- Native prompts: `sequential`, `hierarchical`.
- Native tasks in `run.py`: `gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`, `arc_challenge`, `mbppplus`, `humanevalplus`, `medqa`.
- Native model choices in current `run.py`: `Qwen/Qwen3-4B` and `Qwen/Qwen3-14B`.
- Native GSM8K is immediately runnable through Hugging Face datasets.
- SVAMP is not native. A fair SVAMP-like row requires either adding a local loader/wrapper outside the vendor repo or mapping LatentWire `data/svamp_eval_70.jsonl` into the same item schema consumed by `BaselineMethod`, `TextMASMethod`, and `LatentMASMethod`.
- Native `run.py` prints traces and one final JSON summary to stdout; it does not write structured per-example telemetry files. For paper-grade comparison, we need a wrapper that captures per-example predictions, agent traces, token counts, latency, and cost fields as JSONL sidecars.

## Fair Comparison Ladder

The minimum matched ladder should use the same backbone, same examples, same answer parser, same generation limits, and same telemetry schema:

1. Single-agent baseline: LatentMAS `baseline`.
2. Text multi-agent baseline: LatentMAS `text_mas` with `sequential` and optionally `hierarchical`.
3. LatentMAS: LatentMAS `latent_mas` with a small latent-step sweep.
4. LatentWire target-alone control: our existing target model without imported route.
5. LatentWire strict route-selection method: same examples and decode limits.
6. LatentWire ablations: no route, random route, protected-frontier-only, tokenizer remap-only, target repair-only, and stacked method.

Matched telemetry required for every row:

- `accuracy`: exact-match after the same answer normalization.
- `examples`: number evaluated and dataset split fingerprint.
- `input_tokens`: prompt tokens per example and aggregate.
- `output_tokens`: generated text tokens per example and aggregate.
- `latent_steps`: requested latent steps and actual latent memory length when available.
- `latency_sec`: per-example latency and aggregate wall-clock.
- `tokens_per_sec`: generated-token throughput and total-token throughput.
- `traceability`: number of agent trace records, retained prompt text hashes, output text hashes, and final prediction provenance.
- `route_help`: examples changed from incorrect baseline to correct method.
- `route_harm`: examples changed from correct baseline to incorrect method.
- `costs`: model name, device, dtype, backend, max-new-tokens, batch size, and estimated decode-token cost.
- `failure_tags`: parse failure, arithmetic failure, over-refinement, route mismatch, tokenizer/span mismatch, or latency timeout.

## Exact Runnable Native GSM Commands

These commands run LatentMAS native GSM8K smoke rows without changing the vendor repo. They write logs outside `references/repos/`.

```bash
mkdir -p results/latentmas_competitor_20260421

cd references/repos/LatentMAS

PYTHONPATH=. HF_HOME=/Users/sujeethjinesh/Desktop/LatentWire/.hf_home \
  /Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python run.py \
  --method baseline \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --split test \
  --max_samples 10 \
  --max_new_tokens 512 \
  --generate_bs 1 \
  --device cpu \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  2>&1 | tee /Users/sujeethjinesh/Desktop/LatentWire/results/latentmas_competitor_20260421/gsm10_baseline_qwen3_4b.log

PYTHONPATH=. HF_HOME=/Users/sujeethjinesh/Desktop/LatentWire/.hf_home \
  /Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python run.py \
  --method text_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --split test \
  --max_samples 10 \
  --max_new_tokens 512 \
  --generate_bs 1 \
  --device cpu \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  2>&1 | tee /Users/sujeethjinesh/Desktop/LatentWire/results/latentmas_competitor_20260421/gsm10_textmas_seq_qwen3_4b.log

PYTHONPATH=. HF_HOME=/Users/sujeethjinesh/Desktop/LatentWire/.hf_home \
  /Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --split test \
  --max_samples 10 \
  --max_new_tokens 512 \
  --latent_steps 20 \
  --latent_space_realign \
  --generate_bs 1 \
  --device cpu \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42 \
  2>&1 | tee /Users/sujeethjinesh/Desktop/LatentWire/results/latentmas_competitor_20260421/gsm10_latentmas_seq_l20_qwen3_4b.log
```

Notes:

- These are CPU-safe smoke commands. They may be slow with `Qwen/Qwen3-4B`; use GPU/vLLM for real rows.
- The current LatentMAS `run.py` always samples with `do_sample=True` in the HF path. If `temperature=0.0` is rejected by the installed `transformers` version, use `--temperature 0.01 --top_p 1.0` and record the seed.
- Native logs need a parser before they are paper-grade telemetry.

## Exact SVAMP-Compatible Wrapper Plan

Do not mutate `references/repos/LatentMAS` for the first fair run. Add a LatentWire-side wrapper next:

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python scripts/run_latentmas_competitor_eval.py \
  --latentmas-root references/repos/LatentMAS \
  --method baseline \
  --model-name Qwen/Qwen3-4B \
  --eval-file data/svamp_eval_70.jsonl \
  --limit 10 \
  --max-new-tokens 512 \
  --device cpu \
  --temperature 0.01 \
  --top-p 1.0 \
  --seed 42 \
  --prediction-output results/latentmas_competitor_20260421/svamp10_baseline_qwen3_4b.jsonl

/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python scripts/run_latentmas_competitor_eval.py \
  --latentmas-root references/repos/LatentMAS \
  --method text_mas \
  --prompt sequential \
  --model-name Qwen/Qwen3-4B \
  --eval-file data/svamp_eval_70.jsonl \
  --limit 10 \
  --max-new-tokens 512 \
  --device cpu \
  --temperature 0.01 \
  --top-p 1.0 \
  --seed 42 \
  --prediction-output results/latentmas_competitor_20260421/svamp10_textmas_seq_qwen3_4b.jsonl

/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python scripts/run_latentmas_competitor_eval.py \
  --latentmas-root references/repos/LatentMAS \
  --method latent_mas \
  --prompt sequential \
  --model-name Qwen/Qwen3-4B \
  --eval-file data/svamp_eval_70.jsonl \
  --limit 10 \
  --max-new-tokens 512 \
  --latent-steps 20 \
  --latent-space-realign \
  --device cpu \
  --temperature 0.01 \
  --top-p 1.0 \
  --seed 42 \
  --prediction-output results/latentmas_competitor_20260421/svamp10_latentmas_seq_l20_qwen3_4b.jsonl
```

Wrapper requirements:

- Import LatentMAS methods from `references/repos/LatentMAS` without editing vendor code.
- Convert LatentWire JSONL rows to LatentMAS item dicts with `question`, `solution`, and normalized `gold`.
- Use the same answer extractor as LatentWire competitor rows, or explicitly log parser version when using LatentMAS `extract_gsm8k_answer`.
- Write one prediction JSON object per example.
- Write a `.meta.json` sidecar with aggregate telemetry and environment.
- Retain agent traces in compact hashable form: prompt hash, output hash, token counts, method, prompt mode, latent steps, and correctness.

## Exact Matched LatentWire Commands

Once the LatentMAS smoke files exist, run matched LatentWire rows on the same example limits and model family:

```bash
mkdir -p results/latentmas_competitor_20260421

/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_eval_70.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --limit 10 \
  --prediction-output results/latentmas_competitor_20260421/latentwire_gsm10_target_alone_qwen3_06b.jsonl

/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/svamp_eval_70.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --limit 10 \
  --prediction-output results/latentmas_competitor_20260421/latentwire_svamp10_target_alone_qwen3_06b.jsonl
```

For the actual LatentWire strict-route rows, use the current best route-selection runner and require the same schema fields as the target-alone controls. If the best route runner is not yet stabilized for this comparison, do not claim a head-to-head win; report the gap as a blocker.

## Ablations Inspired By LatentMAS

- Same-model upper bound: run LatentMAS on one Qwen backbone to estimate how much of the gain comes from latent communication without cross-model mismatch.
- Cross-model stress: compare LatentWire on Qwen2.5-to-Qwen3 against LatentMAS same-model rows to separate cross-model alignment cost from latent-message benefit.
- Latent-step sweep: `latent_steps in {0, 5, 10, 20, 40, 80}` with accuracy/latency curves.
- Realignment control: LatentMAS `--latent_space_realign` on/off; this maps to LatentWire transport-plus-repair and gives a clean competitor ablation.
- Text bottleneck control: `text_mas` vs `latent_mas`, matched prompt topology, to quantify serialization loss.
- Traceability control: compare final-answer-only metrics with route help/harm and agent-trace provenance so the result remains interpretable.
- Cost control: report output tokens, prompt tokens, latent steps, and wall-clock separately; do not collapse latent steps into "free tokens".

## Decision For The Paper

LatentMAS is a direct and important competitor, but it should not replace the current positive-method thesis. It gives us a strong same-model latent collaboration ceiling. LatentWire must win on a different claim: cross-model communication under tokenizer/cache/space mismatch with interpretable route selection and repair. The fair paper framing is:

- LatentMAS: same-model latent collaboration competitor.
- C2C/KVComm/KVPress/KVzip/LLMLingua: adjacent cache, communication, compression, and prompt baselines.
- LatentWire: cross-model route selection plus target repair with explicit help/harm telemetry.

The next implementation step is a LatentWire-side `scripts/run_latentmas_competitor_eval.py` wrapper, not vendor repo edits.
