# Source Likelihood Sketch Collection

- date: `2026-04-27`
- status: `source_likelihood_sketch_collected`
- source model: `Qwen/Qwen3-0.6B`
- candidate text field: `normalized_prediction`
- continuation template: `Answer: {text}`
- git commit: `be58c2dcd028857bdaec6d6bcaf619507da63aab`
- eval file: `results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl`
- eval file sha256: `1bfbb11cb8ea958e73d1eb8260f1152dadb7f4c27c1c1dce0392ccafb08f4b14`
- output JSONL: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/slots_only.jsonl`
- output JSONL sha256: `cc3de02212a06145cf4dfa1f6d57ba0779c9f191eae5ef61ecf1fd64065260c1`
- rows: `12`
- ordered IDs sha256: `8e903c697457473261443f473ce7744fca6de0f92f2de074b1f6f3c5e59fe25d`
- resume: `True`
- skipped existing: `0`
- device: `cpu`
- dtype: `float32`

## Command

```bash
scripts/collect_source_likelihood_sketch.py --source-model Qwen/Qwen3-0.6B --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl --candidate target=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/slots_only/target.jsonl,method=target_alone --candidate text=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/slots_only/text.jsonl,method=slots_only_text_candidate --candidate source=path=/Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/slots_only/source.jsonl,method=slots_only_source_candidate --reference-label target --candidate-text-field normalized_prediction --continuation-template 'Answer: {text}' --device cpu --dtype float32 --resume --output-jsonl /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/slots_only.jsonl --output-md /Users/sujeethjinesh/Desktop/LatentWire/results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/slots_only.md
```

## Candidate Inputs

- `target`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/slots_only/target.jsonl` method `target_alone` sha256 `731eb27d51ca9e5de3d4c2a7a040a8796d4fa25ac845cfcaae317489fb986efe`
- `text`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/slots_only/text.jsonl` method `slots_only_text_candidate` sha256 `8d0009fddfcffc34988727405187bcd789eecac5a019cf99692a5e81ff879f61`
- `source`: `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools/slots_only/source.jsonl` method `slots_only_source_candidate` sha256 `467fd0a1f3dda3e65621fbd1dc99b67d6381cad6c58280e7e53d4a108b02d28f`

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
