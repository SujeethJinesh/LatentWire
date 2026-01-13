#!/bin/bash
#SBATCH --job-name=reasoning_final
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/reasoning_final_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/reasoning_final_%j.err

# =============================================================================
# REASONING-FOCUSED FINAL EXPERIMENTS (Publication Ready)
# =============================================================================
# This is the ONE SHOT script for MLSys 2025 - REASONING FOCUS
#
# REASONING BENCHMARKS:
# - ARC-Easy: Elementary science questions (4-way MC)
# - WinoGrande: Common sense pronoun resolution (2-way)
# - HellaSwag: Situational reasoning (4-way MC)
# - BoolQ: Yes/No reading comprehension (2-way)
#
# GPU WORKLOAD:
# - GPU 0: Standard bridge (hail mary) + baselines (zero-shot, few-shot, DoRA, prompt tuning, linear probe)
# - GPU 1: Novel bridges + token capacity + latency/memory/throughput benchmarks + batched latency
#
# EXPERIMENTS:
# 1. Standard bridge on all reasoning tasks (3 seeds) via hail_mary
# 2. Zero-shot baseline on all 4 reasoning datasets (arc_easy, winogrande, hellaswag, boolq)
# 3. Few-shot baseline (4-shot) on arc_easy, winogrande, hellaswag, boolq
# 4. DoRA fine-tuning baseline on arc_easy (SOTA PEFT, 2024)
# 5. Prompt tuning baseline on arc_easy (proves sender helps)
# 6. Linear probe baseline (layer 31) on all reasoning datasets
# 7. Novel bridges: flow_matching, optimal_transport, moe (on arc_easy)
# 8. Token capacity: 4, 8, 16, 32 tokens (on arc_easy)
# 9. Latency/memory/throughput benchmarks
# 10. Batched latency benchmark (batch sizes: 1, 4, 8, 16) - CRITICAL FOR MLSys
#
# Estimated runtime: ~10-11 hours on 2 H100 GPUs
#
# Submit with: sbatch telepathy/submit_reasoning_final.slurm
# =============================================================================

WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "REASONING-FOCUSED FINAL EXPERIMENTS (Publication Ready)"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "REASONING BENCHMARKS:"
echo "  - ARC-Easy (science)"
echo "  - WinoGrande (commonsense)"
echo "  - HellaSwag (situational)"
echo "  - BoolQ (reading comprehension)"
echo ""
echo "GPU ALLOCATION:"
echo "  - GPU 0: Standard bridge (hail mary) + baselines (zero-shot, few-shot, DoRA, prompt tuning, linear probe)"
echo "  - GPU 1: Novel bridges + token capacity + latency/memory/throughput/batched benchmarks"
echo "=============================================================="

export PYTHONPATH=.
export HF_HOME="/projects/m000066/sujinesh/.cache/huggingface"
export TRANSFORMERS_CACHE="/projects/m000066/sujinesh/.cache/huggingface/transformers"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/reasoning_final_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/standard_bridge"
mkdir -p "$OUTPUT_DIR/novel_bridges"
mkdir -p "$OUTPUT_DIR/baselines"
mkdir -p "$OUTPUT_DIR/benchmarks"

# Git setup
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"
git pull || true

echo ""
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

SEEDS="42 123 456"
REASONING_DATASETS="arc_easy winogrande hellaswag boolq"

# =============================================================================
# GPU 0: STANDARD BRIDGE + BASELINES (zero-shot, few-shot, DoRA, prompt tuning, linear probe)
# =============================================================================
run_gpu0() {
    echo "[GPU 0] Starting: Standard Bridge + Baselines"

    # -----------------------------------------------------------------
    # HAIL MARY EVALUATION (Standard bridge on reasoning benchmarks)
    # -----------------------------------------------------------------
    echo "[GPU 0] Running Hail Mary Reasoning Evaluation"

    python telepathy/run_enhanced_paper_evaluation.py \
        --output_dir "${OUTPUT_DIR}/standard_bridge" \
        --gpu 0 \
        --hail_mary_mode \
        --fast_mode \
        --run_diagnostics \
        --log_level INFO \
        2>&1 | tee "${OUTPUT_DIR}/standard_bridge/hail_mary.log" || true

    echo "[GPU 0] Hail mary evaluation completed"

    # -----------------------------------------------------------------
    # ZERO-SHOT BASELINE on all 4 reasoning datasets
    # Critical comparison: how do raw LLMs perform without any training?
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 0] Running Zero-Shot Baseline on all reasoning datasets"

    for DATASET in arc_easy winogrande hellaswag boolq; do
        echo "[GPU 0] Zero-shot on $DATASET"
        python telepathy/run_baselines.py \
            --baseline zeroshot \
            --dataset $DATASET \
            --gpu 0 \
            --max_samples 200 \
            --output_dir "${OUTPUT_DIR}/baselines/zeroshot" \
            2>&1 | tee "${OUTPUT_DIR}/baselines/zeroshot_${DATASET}.log" || true
    done

    # -----------------------------------------------------------------
    # FEW-SHOT BASELINE (4-shot) on all reasoning datasets
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 0] Running Few-Shot Baseline (4-shot) on reasoning datasets"

    for DATASET in arc_easy winogrande hellaswag boolq; do
        echo "[GPU 0] Few-shot (4-shot) on $DATASET"
        python telepathy/run_baselines.py \
            --baseline fewshot \
            --dataset $DATASET \
            --shots 4 \
            --seeds 42 123 456 \
            --gpu 0 \
            --max_samples 200 \
            --output_dir "${OUTPUT_DIR}/baselines/fewshot" \
            2>&1 | tee "${OUTPUT_DIR}/baselines/fewshot_${DATASET}.log" || true
    done

    # -----------------------------------------------------------------
    # DORA FINE-TUNING BASELINE on ARC-Easy
    # DoRA = Weight-Decomposed LoRA, SOTA PEFT method (2024)
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 0] Running DoRA Baseline (rank=8) on ARC-Easy"

    python telepathy/run_baselines.py \
        --baseline dora \
        --dataset arc_easy \
        --rank 8 \
        --epochs 3 \
        --seeds 42 123 456 \
        --gpu 0 \
        --max_samples 200 \
        --max_train_samples 2000 \
        --output_dir "${OUTPUT_DIR}/baselines/dora" \
        2>&1 | tee "${OUTPUT_DIR}/baselines/dora_arc_easy.log" || true

    # -----------------------------------------------------------------
    # PROMPT TUNING BASELINE on ARC-Easy
    # Proves that Llama (sender) actually helps - receiver alone can't match bridge
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 0] Running Prompt Tuning Baseline (8 tokens) on ARC-Easy"

    python telepathy/run_baselines.py \
        --baseline prompt_tuning \
        --dataset arc_easy \
        --soft_tokens 8 \
        --steps 1500 \
        --seeds 42 123 456 \
        --gpu 0 \
        --max_samples 200 \
        --output_dir "${OUTPUT_DIR}/baselines/prompt_tuning" \
        2>&1 | tee "${OUTPUT_DIR}/baselines/prompt_tuning_arc_easy.log" || true

    # -----------------------------------------------------------------
    # LINEAR PROBE BASELINE (Layer 31 - same as bridge)
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 0] Running Linear Probe Baseline (layer 31)"

    python telepathy/linear_probe_baseline.py \
        --datasets arc_easy winogrande hellaswag boolq \
        --layers 31 \
        --seeds 42 123 456 \
        --output_dir "${OUTPUT_DIR}/baselines/linear_probe" \
        --device cuda:0 \
        2>&1 | tee "${OUTPUT_DIR}/baselines/linear_probe.log" || true

    echo "[GPU 0] All evaluations completed"
}

# =============================================================================
# GPU 1: NOVEL BRIDGES + TOKEN CAPACITY + LATENCY/MEMORY/THROUGHPUT/BATCHED BENCHMARKS
# =============================================================================
run_gpu1() {
    echo "[GPU 1] Starting: Novel Bridges + Token Capacity + All Benchmarks"

    # -----------------------------------------------------------------
    # NOVEL BRIDGES on ARC-Easy (simplest reasoning task)
    # 3 bridges: flow_matching, optimal_transport, moe (Mixtral/DeepSeek-style)
    # -----------------------------------------------------------------
    NOVEL_BRIDGES="flow_matching optimal_transport moe"

    for BRIDGE in $NOVEL_BRIDGES; do
        echo ""
        echo "[GPU 1] Testing $BRIDGE on ARC-Easy"

        python telepathy/train_telepathy.py \
            --dataset arc_easy \
            --bridge_type $BRIDGE \
            --soft_tokens 8 \
            --steps 1500 \
            --eval_every 200 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/novel_bridges/${BRIDGE}_arc_easy" \
            --gpu 1 \
            2>&1 | tee "${OUTPUT_DIR}/novel_bridges/${BRIDGE}_arc_easy.log" || true
    done

    # -----------------------------------------------------------------
    # TOKEN CAPACITY EXPERIMENT (Does more tokens help reasoning?)
    # Full ablation curve: 4, 8, 16, 32 tokens
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 1] Token Capacity Scaling on Reasoning (4, 8, 16, 32 tokens)"

    for TOKENS in 4 8 16 32; do
        echo "[GPU 1] Testing $TOKENS soft tokens on ARC-Easy"
        python telepathy/train_telepathy.py \
            --dataset arc_easy \
            --bridge_type standard \
            --soft_tokens $TOKENS \
            --steps 1500 \
            --eval_every 200 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/novel_bridges/token_capacity_${TOKENS}" \
            --gpu 1 \
            2>&1 | tee "${OUTPUT_DIR}/novel_bridges/token_capacity_${TOKENS}.log" || true
    done

    # -----------------------------------------------------------------
    # LATENCY BENCHMARKS
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 1] Running Latency Benchmarks"

    # Find a checkpoint to use for latency benchmark (from token_capacity_8)
    CHECKPOINT="${OUTPUT_DIR}/novel_bridges/token_capacity_8/bridge_arc_easy.pt"

    # Wait a bit for checkpoint to be written
    sleep 5

    if [ -f "$CHECKPOINT" ]; then
        echo "[GPU 1] Running latency benchmark with checkpoint: $CHECKPOINT"
        python telepathy/run_benchmarks.py \
            --benchmark latency \
            --checkpoint "$CHECKPOINT" \
            --soft_tokens 8 \
            --num_trials 50 \
            --warmup 5 \
            --gpu 1 \
            --output_dir "${OUTPUT_DIR}/benchmarks" \
            2>&1 | tee "${OUTPUT_DIR}/benchmarks/latency.log" || true
    else
        echo "[GPU 1] No checkpoint found, running latency benchmark without bridge"
        python telepathy/run_benchmarks.py \
            --benchmark latency \
            --soft_tokens 8 \
            --num_trials 50 \
            --warmup 5 \
            --gpu 1 \
            --output_dir "${OUTPUT_DIR}/benchmarks" \
            2>&1 | tee "${OUTPUT_DIR}/benchmarks/latency.log" || true
    fi

    # -----------------------------------------------------------------
    # BATCHED LATENCY BENCHMARK (Critical for MLSys paper)
    # Tests throughput at different batch sizes: 1, 4, 8, 16
    # -----------------------------------------------------------------
    echo ""
    echo "[GPU 1] Running Batched Latency Benchmark (batch sizes: 1, 4, 8, 16)"
    python telepathy/run_benchmarks.py \
        --benchmark batched \
        --batch_sizes 1 4 8 16 \
        --soft_tokens 8 \
        --num_samples 100 \
        --num_runs 5 \
        --warmup 2 \
        --gpu 1 \
        --output_dir "${OUTPUT_DIR}/benchmarks" \
        2>&1 | tee "${OUTPUT_DIR}/benchmarks/batched_latency.log" || true

    # Memory benchmark
    echo ""
    echo "[GPU 1] Running Memory Benchmark"
    python telepathy/run_benchmarks.py \
        --benchmark memory \
        --gpu 1 \
        --output_dir "${OUTPUT_DIR}/benchmarks" \
        2>&1 | tee "${OUTPUT_DIR}/benchmarks/memory.log" || true

    # Throughput benchmark
    echo ""
    echo "[GPU 1] Running Throughput Benchmark"
    if [ -f "$CHECKPOINT" ]; then
        python telepathy/run_benchmarks.py \
            --benchmark throughput \
            --checkpoint "$CHECKPOINT" \
            --soft_tokens 8 \
            --num_samples 100 \
            --gpu 1 \
            --output_dir "${OUTPUT_DIR}/benchmarks" \
            2>&1 | tee "${OUTPUT_DIR}/benchmarks/throughput.log" || true
    else
        python telepathy/run_benchmarks.py \
            --benchmark throughput \
            --soft_tokens 8 \
            --num_samples 100 \
            --gpu 1 \
            --output_dir "${OUTPUT_DIR}/benchmarks" \
            2>&1 | tee "${OUTPUT_DIR}/benchmarks/throughput.log" || true
    fi

    echo "[GPU 1] All experiments completed"
}

# =============================================================================
# RUN BOTH GPUS IN PARALLEL
# =============================================================================
echo ""
echo "Starting parallel execution on 2 GPUs..."
echo ""

run_gpu0 &
GPU0_PID=$!

run_gpu1 &
GPU1_PID=$!

echo "GPU 0 PID: $GPU0_PID (Standard Bridge + Baselines)"
echo "GPU 1 PID: $GPU1_PID (Novel Bridges + Benchmarks)"
echo ""

wait $GPU0_PID
echo "GPU 0 finished"

wait $GPU1_PID
echo "GPU 1 finished"

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
echo ""
echo "=============================================================="
echo "AGGREGATING REASONING RESULTS"
echo "=============================================================="

python -c "
import json
from pathlib import Path
import os

output_dir = Path('${OUTPUT_DIR}')

print('='*70)
print('REASONING BENCHMARK RESULTS (Publication Ready)')
print('='*70)

# Standard bridge results (from hail mary)
hail_mary_results = list(output_dir.glob('standard_bridge/run_*/enhanced_evaluation_results.json'))
if hail_mary_results:
    with open(hail_mary_results[0]) as f:
        results = json.load(f)

    print()
    print('STANDARD BRIDGE (3 seeds):')
    print('-'*40)

    for dataset in ['arc_easy', 'winogrande', 'hellaswag', 'boolq']:
        if 'classification' in results and 'aggregated' in results['classification']:
            for method in ['bridge_8', 'zeroshot_llama', 'zeroshot_mistral', 'linear_probe']:
                stats = results['classification']['aggregated'].get(method, {}).get(dataset, {})
                if stats and 'mean' in stats:
                    print(f'  {dataset:12} | {method:18} | {stats[\"mean\"]:.1f}% +/- {stats.get(\"std\", 0):.1f}%')

# Zero-shot baseline results (NEW)
print()
print('ZERO-SHOT BASELINE (no training):')
print('-'*40)

for zeroshot_file in sorted(output_dir.glob('baselines/zeroshot/*.json')):
    with open(zeroshot_file) as f:
        data = json.load(f)
    if 'results' in data:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        for model, stats in data['results'].items():
            if isinstance(stats, dict) and 'accuracy' in stats:
                print(f'  {dataset:12} | {model:10} | {stats[\"accuracy\"]:.1f}%')

# Few-shot baseline results
print()
print('FEW-SHOT BASELINE (4-shot):')
print('-'*40)

for fewshot_file in sorted(output_dir.glob('baselines/fewshot/*.json')):
    with open(fewshot_file) as f:
        data = json.load(f)
    if 'results' in data:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        for model, stats in data['results'].items():
            if isinstance(stats, dict) and 'mean_accuracy' in stats:
                print(f'  {dataset:12} | {model:10} | {stats[\"mean_accuracy\"]:.1f}% +/- {stats.get(\"std_accuracy\", 0):.1f}%')

# DoRA baseline results
print()
print('DORA BASELINE (rank=8, SOTA 2024):')
print('-'*40)

for dora_file in sorted(output_dir.glob('baselines/dora/*.json')):
    with open(dora_file) as f:
        data = json.load(f)
    if 'results' in data and 'summary' in data['results']:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        summary = data['results']['summary']
        print(f'  {dataset:12} | DoRA r=8   | {summary[\"mean_accuracy\"]:.1f}% +/- {summary.get(\"std_accuracy\", 0):.1f}%')

# Prompt tuning baseline results
print()
print('PROMPT TUNING BASELINE (8 tokens, no sender):')
print('-'*40)

for pt_file in sorted(output_dir.glob('baselines/prompt_tuning/*.json')):
    with open(pt_file) as f:
        data = json.load(f)
    if 'results' in data and 'summary' in data['results']:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        summary = data['results']['summary']
        print(f'  {dataset:12} | Prompt Tuning | {summary[\"mean_accuracy\"]:.1f}% +/- {summary.get(\"std_accuracy\", 0):.1f}%')

# Novel bridge results
print()
print('NOVEL BRIDGES (ARC-Easy):')
print('-'*40)

for bridge_dir in sorted(output_dir.glob('novel_bridges/*_arc_easy')):
    for json_file in bridge_dir.glob('*_results.json'):
        with open(json_file) as f:
            data = json.load(f)
        if 'final_results' in data:
            acc = data['final_results'].get('accuracy', 0)
            bridge_name = bridge_dir.name.replace('_arc_easy', '')
            print(f'  {bridge_name:25} | {acc:.1f}%')

# Token capacity results
print()
print('TOKEN CAPACITY (ARC-Easy):')
print('-'*40)

for cap_dir in sorted(output_dir.glob('novel_bridges/token_capacity_*')):
    for json_file in cap_dir.glob('*_results.json'):
        with open(json_file) as f:
            data = json.load(f)
        if 'final_results' in data:
            acc = data['final_results'].get('accuracy', 0)
            tokens = cap_dir.name.split('_')[-1]
            print(f'  {tokens} tokens | {acc:.1f}%')

# Latency benchmark results
print()
print('LATENCY BENCHMARKS:')
print('-'*40)

for bench_file in sorted(output_dir.glob('benchmarks/latency_*.json')):
    with open(bench_file) as f:
        data = json.load(f)
    if 'results' in data:
        results = data['results']
        if 'direct_text' in results:
            print(f'  Direct Text:  {results[\"direct_text\"][\"avg_ms\"]:.1f} ms')
        if 'text_relay' in results:
            print(f'  Text-Relay:   {results[\"text_relay\"][\"avg_ms\"]:.1f} ms')
        if 'bridge' in results:
            print(f'  Bridge:       {results[\"bridge\"][\"avg_ms\"]:.1f} ms')
            if 'text_relay' in results:
                speedup = results['text_relay']['avg_ms'] / results['bridge']['avg_ms']
                print(f'  Speedup:      {speedup:.1f}x vs Text-Relay')

# Batched latency benchmark results (NEW - Critical for MLSys)
print()
print('BATCHED LATENCY (samples/sec by batch size):')
print('-'*40)

for bench_file in sorted(output_dir.glob('benchmarks/batched_*.json')):
    with open(bench_file) as f:
        data = json.load(f)
    if 'results' in data:
        results = data['results']
        print(f'  {\"Batch\":<8} {\"Bridge\":<15} {\"Direct\":<15}')
        print(f'  {\"-----\":<8} {\"------\":<15} {\"------\":<15}')
        if 'bridge' in results and 'direct_mistral' in results:
            for i, bridge_res in enumerate(results['bridge']):
                bs = bridge_res['batch_size']
                bridge_tp = bridge_res['throughput_samples_per_s']
                direct_tp = results['direct_mistral'][i]['throughput_samples_per_s']
                print(f'  {bs:<8} {bridge_tp:<15.1f} {direct_tp:<15.1f}')

print('='*70)

# Save combined results
combined = {
    'hail_mary': str(hail_mary_results[0]) if hail_mary_results else None,
    'novel_bridges': {},
    'token_capacity': {},
    'baselines': {
        'zeroshot': {},
        'fewshot': {},
        'dora': {},
        'prompt_tuning': {}
    },
    'benchmarks': {
        'latency': {},
        'batched': {},
        'memory': {},
        'throughput': {}
    }
}
with open(output_dir / 'reasoning_results_summary.json', 'w') as f:
    json.dump(combined, f, indent=2)
" 2>/dev/null || echo "Could not aggregate (some experiments may have failed)"

# =============================================================================
# GIT COMMIT
# =============================================================================
echo ""
echo "Saving to git..."

git add "${OUTPUT_DIR}/" 2>/dev/null || true

COMMIT_MSG="results: REASONING-FOCUSED final experiments (job $SLURM_JOB_ID)

REASONING BENCHMARKS:
- ARC-Easy, WinoGrande, HellaSwag, BoolQ

EXPERIMENTS:
1. Standard bridge (3 seeds) via hail_mary evaluation
2. Zero-shot baseline on all 4 reasoning datasets (Llama + Mistral)
3. Few-shot baseline (4-shot) on arc_easy, winogrande, hellaswag, boolq
4. DoRA baseline (rank=8) on arc_easy (SOTA PEFT 2024)
5. Prompt tuning baseline (8 tokens, no sender) on arc_easy
6. Linear probe baseline (layer 31) on all reasoning datasets
7. Novel bridges (3):
   - Flow Matching, Optimal Transport, MoE (Mixtral/DeepSeek-style)
8. Token capacity scaling (4, 8, 16, 32 tokens)
9. Latency/memory/throughput benchmarks
10. Batched latency benchmark (batch sizes: 1, 4, 8, 16) - Critical for MLSys

KEY QUESTIONS:
- Can novel compression techniques improve reasoning?
- Does more tokens help reasoning?
- What's the latency advantage over text-relay?
- Does the sender (Llama) actually help? (prompt_tuning vs bridge)
- How does throughput scale with batch size?

Runtime: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git commit -m "$COMMIT_MSG" && git push || echo "Git commit/push failed"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "=============================================================="
echo "REASONING EXPERIMENTS COMPLETED (Publication Ready)"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
echo ""
echo "KEY EXPERIMENTS:"
echo "  1. Standard bridge on reasoning? (hail_mary)"
echo "  2. Zero-shot baseline? (baselines/zeroshot/)"
echo "  3. Few-shot baseline (4-shot)? (baselines/fewshot/)"
echo "  4. DoRA baseline (SOTA 2024)? (baselines/dora/)"
echo "  5. Prompt tuning (does sender help)? (baselines/prompt_tuning/)"
echo "  6. Linear probe upper bound? (baselines/linear_probe/)"
echo "  7. Novel bridges help? (novel_bridges/)"
echo "  8. More tokens help? (token_capacity: 4, 8, 16, 32)"
echo "  9. Latency advantage? (benchmarks/latency)"
echo " 10. Batched throughput scaling? (benchmarks/batched) - MLSys Critical"
echo ""
echo "Results: ${OUTPUT_DIR}"
echo "=============================================================="
