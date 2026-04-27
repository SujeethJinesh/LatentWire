# Source Likelihood Sketch Collection

- date: `2026-04-27`
- status: `source_likelihood_sketch_collected`
- source model: `Qwen/Qwen3-0.6B`
- candidate text field: `normalized_prediction`
- continuation template: `Answer: {text}`
- git commit: `be58c2dcd028857bdaec6d6bcaf619507da63aab`
- eval file: `results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl`
- eval file sha256: `1bfbb11cb8ea958e73d1eb8260f1152dadb7f4c27c1c1dce0392ccafb08f4b14`
- output JSONL: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_only.jsonl`
- output JSONL sha256: `fbc34d474466922f3678f0615e2fab8a88e3f1ee90723279f1d3626267e891a7`
- rows: `12`
- ordered IDs sha256: `8e903c697457473261443f473ce7744fca6de0f92f2de074b1f6f3c5e59fe25d`
- resume: `True`
- skipped existing: `0`
- device: `cpu`
- dtype: `float32`

## Command

```bash
scripts/collect_source_likelihood_sketch.py --source-model Qwen/Qwen3-0.6B --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl --candidate target=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/answer_only/target.jsonl,method=target_alone --candidate text=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/answer_only/text.jsonl,method=text_to_text --candidate source=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/answer_only/source.jsonl,method=answer_only_source_candidate --reference-label target --candidate-text-field normalized_prediction --continuation-template 'Answer: {text}' --device cpu --dtype float32 --resume --output-jsonl /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_only.jsonl --output-md /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_only.md
```

## Candidate Inputs

- `target`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/answer_only/target.jsonl` method `target_alone` sha256 `7ce2f80ad8f98d81611767ff700f7b4ac5670ee8f268d7dd75f745e80f28a38a`
- `text`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/answer_only/text.jsonl` method `text_to_text` sha256 `3b4ee213e6c7b2ff1ff9a2f2226838e9d84792c0eeca2b300bf7605c72d34ce0`
- `source`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/answer_only/source.jsonl` method `answer_only_source_candidate` sha256 `a4151331ff6d17ff763123acf52f97bf52687c40671b742a13b0bc897eab90f7`

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
