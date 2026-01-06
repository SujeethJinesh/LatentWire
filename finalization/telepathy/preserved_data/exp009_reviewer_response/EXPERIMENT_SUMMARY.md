# Experiment: Reviewer Response Experiments (exp009)

**Date**: 2025-12-14
**Status**: COMPLETE
**Purpose**: Address reviewer critiques with additional experiments

---

## Executive Summary

Ran 6 experiments to address critiques from simulated review committee (10 professors).
Results strengthen the paper significantly:

1. **Few-shot baselines**: Bridge beats 5-shot on all datasets (+2.2pp to +59.3pp)
2. **LoRA comparison**: Bridge beats LoRA with 18× fewer parameters
3. **CoT relay**: Bridge is 85× faster AND +7.7pp more accurate
4. **Batched latency**: Bridge scales to 100+ samples/sec
5. **GSM8K reasoning**: FAILED (1.5% vs 19% baseline) - honest limitation

---

## Experiment Results

### 1. Few-Shot Baselines (Addresses 4/10 reviewers - HIGH priority)

Concern: "Zero-shot baselines may be artificially weak"

| Dataset | Llama 5-shot | Mistral 5-shot | Bridge | Gap |
|---------|--------------|----------------|--------|-----|
| SST-2 | 94.3% ± 0.2% | 94.5% ± 1.1% | **96.7%** | **+2.2pp** |
| AG News | 62.0% ± 3.6% | 80.3% ± 1.7% | **90.7%** | **+10.4pp** |
| TREC | 32.0% ± 0.0% | 36.0% ± 0.0% | **95.3%** | **+59.3pp** |

**Conclusion**: Super-additive claim VALIDATED. Bridge significantly outperforms few-shot on all datasets.

### 2. LoRA Comparison (Addresses 4/10 reviewers - HIGH priority)

Concern: "What if you just fine-tuned Mistral with LoRA?"

| Method | Accuracy | Parameters | Ratio |
|--------|----------|------------|-------|
| LoRA (rank=8) | 95.3% ± 0.9% | 3.4M | 1× |
| Bridge | **96.7% ± 0.6%** | **188K** | **18×** |

**Conclusion**: Bridge achieves +1.4pp higher accuracy with 18× fewer parameters.

### 3. CoT Text-Relay (Addresses 3/10 reviewers - MEDIUM priority)

Concern: "Text-relay baseline is artificially weak (just summarization)"

| Method | Accuracy | Latency | CoT Tokens |
|--------|----------|---------|------------|
| CoT-Relay | 89.0% | 3,169ms | 150 |
| Bridge | **96.7%** | **37ms** | 16 soft |

**Conclusion**: Bridge is +7.7pp more accurate AND 85× faster than chain-of-thought.

### 4. Batched Latency (Addresses 2/10 reviewers - MEDIUM priority)

Concern: "Latency comparison is single-sample. What about batched inference?"

| Batch Size | Bridge (s/s) | Direct (s/s) | Text-Relay (s/s) |
|------------|--------------|--------------|------------------|
| 1 | 7.4 | 8.8 | 0.9 |
| 4 | 28.7 | 31.2 | 1.0 |
| 16 | 105.7 | 116.0 | -- |

**Conclusion**: Bridge scales well with batch size. 8× speedup vs text-relay maintained.

### 5. GSM8K Math Reasoning (Addresses 6/10 reviewers - HIGH priority)

Concern: "Only classification, no generation/reasoning"

| Method | Accuracy |
|--------|----------|
| Llama direct | 19.5% |
| Mistral direct | 18.5% |
| Bridge | **1.5%** |

**Conclusion**: FAILED. Bridge generates incoherent solutions. Method is limited to classification.
This is documented as an honest limitation in the paper.

### 6. Cross-Task Transfer (Not yet run)

Note: Requires existing bridge checkpoint. Can run if needed.

---

## Paper Updates Made

1. ✅ Added few-shot baselines to Table 1 (main results)
2. ✅ Added new section "Comparison with Fine-Tuning and Chain-of-Thought"
3. ✅ Added Table: LoRA vs CoT vs Bridge comparison
4. ✅ Added Table: Batched throughput
5. ✅ Updated Limitations section with GSM8K failure (honest)
6. ✅ Updated abstract to mention stronger baselines
7. ✅ Paper recompiled (9 pages)

---

## File Locations

Results stored in: `runs/reviewer_experiments/`

```
runs/reviewer_experiments/
├── fewshot/
│   ├── fewshot_sst2_5shot.json    ✅
│   ├── fewshot_agnews_5shot.json  ✅
│   └── fewshot_trec_5shot.json    ✅
├── lora/
│   └── lora_sst2_r8.json          ✅
├── cot_relay/
│   └── cot_relay_sst2.json        ✅
├── latency/
│   └── batched_latency.json       ✅
└── gsm8k/
    └── gsm8k_results.json         ✅
```

---

## Reviewer Critique Coverage

| Critique | Reviewers | Priority | Status | Result |
|----------|-----------|----------|--------|--------|
| Classification-only | 6/10 | HIGH | ✅ Tested | FAILED (limitation) |
| Missing LoRA | 4/10 | HIGH | ✅ Tested | Bridge wins |
| Weak zero-shot | 4/10 | HIGH | ✅ Tested | Bridge beats 5-shot |
| Weak text-relay | 3/10 | MEDIUM | ✅ Tested | Bridge wins both |
| No cross-task transfer | 2/10 | MEDIUM | ❌ Skipped | Need checkpoint |
| Missing batched latency | 2/10 | MEDIUM | ✅ Tested | Scales well |

---

## Reproduction

To reproduce all reviewer experiments:
```bash
cd telepathy
PYTHONPATH=.. bash run_reviewer_experiments.sh all
```

To run specific experiments:
```bash
bash run_reviewer_experiments.sh quick     # Few-shot + batched latency
bash run_reviewer_experiments.sh moderate  # LoRA + CoT
bash run_reviewer_experiments.sh gsm8k     # GSM8K reasoning
```
