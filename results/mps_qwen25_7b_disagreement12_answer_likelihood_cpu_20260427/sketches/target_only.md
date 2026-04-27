# Source Likelihood Sketch Collection

- date: `2026-04-27`
- status: `source_likelihood_sketch_collected`
- source model: `Qwen/Qwen3-0.6B`
- candidate text field: `normalized_prediction`
- continuation template: `Answer: {text}`
- git commit: `be58c2dcd028857bdaec6d6bcaf619507da63aab`
- eval file: `results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl`
- eval file sha256: `1bfbb11cb8ea958e73d1eb8260f1152dadb7f4c27c1c1dce0392ccafb08f4b14`
- output JSONL: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/target_only.jsonl`
- output JSONL sha256: `3b73142dabaeee11f11cfc2b335d644bdb8a53fd2937835bcd57ffce450a8c79`
- rows: `12`
- ordered IDs sha256: `8e903c697457473261443f473ce7744fca6de0f92f2de074b1f6f3c5e59fe25d`
- resume: `True`
- skipped existing: `0`
- device: `cpu`
- dtype: `float32`

## Command

```bash
scripts/collect_source_likelihood_sketch.py --source-model Qwen/Qwen3-0.6B --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl --candidate target=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/target_only/target.jsonl,method=target_alone --candidate text=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/target_only/text.jsonl,method=target_only_text_candidate --candidate source=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/target_only/source.jsonl,method=target_only_source_candidate --reference-label target --candidate-text-field normalized_prediction --continuation-template 'Answer: {text}' --device cpu --dtype float32 --resume --output-jsonl /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/target_only.jsonl --output-md /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/target_only.md
```

## Candidate Inputs

- `target`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/target_only/target.jsonl` method `target_alone` sha256 `327c70019597e11ef23f5f29cf9c42c7c7aba6cdc5b4d9ec1ee4dfd7863a90eb`
- `text`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/target_only/text.jsonl` method `target_only_text_candidate` sha256 `c442d5fbee477c0f4fad37bdb48c3980b075dbb649f205743d460acef576591d`
- `source`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/target_only/source.jsonl` method `target_only_source_candidate` sha256 `037bfe7d6bdbd4bdb27bce75fe7d24a4d3bc9ef926f872c153f561ae4f012e8a`

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
