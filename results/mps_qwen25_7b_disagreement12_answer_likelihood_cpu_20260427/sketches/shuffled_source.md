# Source Likelihood Sketch Collection

- date: `2026-04-27`
- status: `source_likelihood_sketch_collected`
- source model: `Qwen/Qwen3-0.6B`
- candidate text field: `normalized_prediction`
- continuation template: `Answer: {text}`
- git commit: `be58c2dcd028857bdaec6d6bcaf619507da63aab`
- eval file: `results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl`
- eval file sha256: `1bfbb11cb8ea958e73d1eb8260f1152dadb7f4c27c1c1dce0392ccafb08f4b14`
- output JSONL: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/shuffled_source.jsonl`
- output JSONL sha256: `5842b4c27a96cbb30414782427bc0d0218864e411b73f0aa718788a4719c455f`
- rows: `12`
- ordered IDs sha256: `8e903c697457473261443f473ce7744fca6de0f92f2de074b1f6f3c5e59fe25d`
- resume: `True`
- skipped existing: `0`
- device: `cpu`
- dtype: `float32`

## Command

```bash
scripts/collect_source_likelihood_sketch.py --source-model Qwen/Qwen3-0.6B --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl --candidate target=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/shuffled_source/target.jsonl,method=target_alone --candidate text=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/shuffled_source/text.jsonl,method=text_to_text --candidate source=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/shuffled_source/source.jsonl,method=shuffled_source_candidate --reference-label target --candidate-text-field normalized_prediction --continuation-template 'Answer: {text}' --device cpu --dtype float32 --resume --output-jsonl /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/shuffled_source.jsonl --output-md /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/shuffled_source.md
```

## Candidate Inputs

- `target`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/shuffled_source/target.jsonl` method `target_alone` sha256 `3a7586f919cca31b2c83a42b0f537e39e4fde299a4d079227ec9998b499d4138`
- `text`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/shuffled_source/text.jsonl` method `text_to_text` sha256 `99269150e93ab2046efc50d5f98d96042264c6d35d3f54c4341d3b01c56f7d37`
- `source`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/shuffled_source/source.jsonl` method `shuffled_source_candidate` sha256 `61b6d4ff318de268439188090533ecf698c411e0a12721c187f34de2ea4c220c`

## Ordered Example IDs

- `14bfbfc94f2c2e7b`
- `2de1549556000830`
- `41cce6c6e6bb0058`
- `4d780f825bb8541c`
- `bd9d8da923981d69`
- `ce08a3a269bf0151`
- `0ee313c160b638a9`
- `561daa750422c0e4`
- `cd5623c80cf95da9`
- `e90d2681e386fb04`
- `ab1e71e8928661d0`
- `daea537474de16ac`

The JSONL contains source-model continuation likelihoods over the candidate answer pool. The downstream gate transmits only a quantized top-label/margin sketch and compares it to source-destroyed controls.
