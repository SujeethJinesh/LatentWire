# Parallel Execution Plan for 4 H100 GPUs

## Executive Summary

This plan optimizes the use of 4 H100 GPUs on the HPC cluster to complete all required experiments within a 24-hour window. The strategy uses SLURM's `srun` command to parallelize independent experiments while respecting memory constraints and model requirements.

## GPU Allocation Strategy

### GPU Assignment by Experiment Type

| GPU | Primary Use | Memory Allocation | Experiment Types |
|-----|------------|------------------|-----------------|
| GPU 0 | Dual-model experiments | 64-80GB | Telepathy bridge training |
| GPU 1 | Dual-model experiments | 64-80GB | Telepathy bridge training |
| GPU 2 | Single-model baselines | 40-50GB | Linear probe, LoRA, direct prompting |
| GPU 3 | Single-model baselines | 40-50GB | Linear probe, LoRA, direct prompting |

### Memory Considerations

- **Dual-model experiments** (Telepathy): Require loading both sender and receiver models
  - Small models (1B/1.5B): 20-30GB per GPU
  - Medium models (7B/8B): 60-70GB per GPU
  - Large models: Not feasible in parallel, run sequentially

- **Single-model experiments**: Only load one model at a time
  - Can run 2-3 experiments per GPU for small models
  - 1 experiment per GPU for medium/large models

## Execution Phases

### Phase 1: Statistical Rigor (6 hours)
**3 seeds × 3 tasks × dual models = 9 experiments**

```bash
# Launch on GPUs 0-1 (dual model experiments)
# Each seed-task combination runs independently
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:0 \
    python telepathy/train_telepathy_sst2.py --seed 42 --output phase1_sst2_s42 &

srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:1 \
    python telepathy/train_telepathy_sst2.py --seed 123 --output phase1_sst2_s123 &

# GPUs 2-3 run baseline preparations
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:2 \
    python telepathy/eval_zeroshot_baselines.py --task sst2 &

srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:3 \
    python telepathy/eval_zeroshot_baselines.py --task agnews &
```

### Phase 2: Linear Probe Baseline (4 hours)
**2 models × 4 layers × 3 tasks = 24 experiments**

```bash
# Distribute across all 4 GPUs
# Each GPU handles 6 experiments (can run in sequence)
for gpu in 0 1 2 3; do
    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:$gpu \
        bash -c "
            for layer in 8 16 24 31; do
                python telepathy/linear_probe_baseline.py \
                    --model meta-llama/Llama-3.1-1B-Instruct \
                    --layer \$layer \
                    --task sst2 \
                    --output probe_llama_L\${layer}_sst2
            done
        " &
done
```

### Phase 3: Fair Baseline Comparisons (6 hours)
**5 baselines × 3 tasks = 15 experiments**

```bash
# GPU 0-1: Telepathy baselines (need dual models)
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:0 \
    python telepathy/train_telepathy_sst2.py --compression 8 &

srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:1 \
    python telepathy/train_telepathy_agnews.py --compression 8 &

# GPU 2: LLMLingua baselines
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:2 \
    bash -c "
        python scripts/run_llmlingua_baseline.sh --task sst2
        python scripts/run_llmlingua_baseline.sh --task agnews
        python scripts/run_llmlingua_baseline.sh --task trec
    " &

# GPU 3: Direct prompting baselines
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:3 \
    bash -c "
        python telepathy/run_llama_baselines.py --task sst2
        python telepathy/run_llama_baselines.py --task agnews
        python telepathy/run_llama_baselines.py --task trec
    " &
```

### Phase 4: Latency Measurements (2 hours)
**4 compression factors × 4 batch sizes = 16 measurements**

```bash
# Distribute latency tests across all GPUs
# Each GPU handles 4 configurations
configs=(
    "4,1" "4,4" "4,8" "4,16"
    "8,1" "8,4" "8,8" "8,16"
    "16,1" "16,4" "16,8" "16,16"
    "32,1" "32,4" "32,8" "32,16"
)

gpu=0
for config in "${configs[@]}"; do
    IFS=',' read -r cf bs <<< "$config"
    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:$gpu \
        python telepathy/benchmark_batched_latency.py \
            --compression_factor $cf \
            --batch_size $bs \
            --output latency_cf${cf}_bs${bs} &

    # Rotate through GPUs
    gpu=$(( (gpu + 1) % 4 ))

    # Wait if all GPUs are busy (every 4 jobs)
    if [ $gpu -eq 0 ]; then
        wait
    fi
done
```

### Phase 5: Generation Task - XSUM (4 hours)
**1 model pair × 500 samples**

```bash
# Use GPUs 0-1 for training, GPUs 2-3 for evaluation
srun --exclusive -N1 -n1 --gpus=2 --gpu-bind=map_gpu:0,1 \
    python telepathy/train_xsum_bridge.py \
        --train_samples 1000 \
        --batch_size 2 \
        --output xsum_bridge &

# Meanwhile, prepare baseline XSUM evaluations on GPUs 2-3
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:2 \
    python telepathy/eval_xsum_bridge.py --baseline llama &

srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:3 \
    python telepathy/eval_xsum_bridge.py --baseline qwen &
```

### Phase 6: Model Size Ablations (2 hours)
**3 model sizes × 1 task = 3 experiments**

```bash
# Run different model sizes on different GPUs based on memory
# Small models on GPU 0
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:0 \
    python telepathy/run_ablation_experiments.py \
        --model_size small \
        --task sst2 \
        --batch_size 4 &

# Medium models on GPUs 1-2 (need more memory)
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:1 \
    python telepathy/run_ablation_experiments.py \
        --model_size medium \
        --task sst2 \
        --batch_size 2 &

# Large models on GPU 3
srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:3 \
    python telepathy/run_ablation_experiments.py \
        --model_size large \
        --task sst2 \
        --batch_size 1 &
```

## Complete SLURM Script

### File: `telepathy/submit_parallel_experiments.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=parallel_experiments
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/parallel_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/parallel_%j.err

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

# Setup environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4  # 16 total threads / 4 GPUs

# Create output directories
OUTPUT_BASE="runs/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"/{phase1,phase2,phase3,phase4,phase5,phase6}

# Function to run phase with timing
run_phase() {
    local phase=$1
    local duration=$2
    echo "Starting Phase $phase at $(date)"
    start_time=$(date +%s)

    case $phase in
        1) run_phase1_parallel ;;
        2) run_phase2_parallel ;;
        3) run_phase3_parallel ;;
        4) run_phase4_parallel ;;
        5) run_phase5_parallel ;;
        6) run_phase6_parallel ;;
    esac

    wait  # Wait for all background jobs in this phase

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Phase $phase completed in $elapsed seconds (expected: $duration hours)"
}

# Phase 1: Statistical Rigor (6 hours)
run_phase1_parallel() {
    local seeds=(42 123 456)
    local tasks=(sst2 agnews trec)
    local gpu=0

    for seed in "${seeds[@]}"; do
        for task in "${tasks[@]}"; do
            # Assign to GPU in round-robin fashion
            srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:$gpu \
                python telepathy/train_telepathy_${task}.py \
                --seed $seed \
                --compression_factor 8 \
                --train_samples 1000 \
                --test_samples -1 \
                --batch_size 4 \
                --output_dir "$OUTPUT_BASE/phase1/${task}_seed${seed}" \
                > "$OUTPUT_BASE/phase1/${task}_seed${seed}.log" 2>&1 &

            gpu=$(( (gpu + 1) % 2 ))  # Use only GPU 0-1 for dual models

            # Start baselines on GPU 2-3
            if [ $gpu -eq 0 ]; then
                srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:2 \
                    python telepathy/eval_zeroshot_baselines.py \
                    --task $task --seed $seed \
                    --output_dir "$OUTPUT_BASE/phase1/baseline_${task}_seed${seed}" \
                    > "$OUTPUT_BASE/phase1/baseline_${task}_seed${seed}.log" 2>&1 &
            fi
        done
    done
}

# Phase 2: Linear Probe (4 hours)
run_phase2_parallel() {
    local models=("meta-llama/Llama-3.1-1B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3")
    local layers=(8 16 24 31)
    local tasks=(sst2 agnews)

    # Create job array for linear probes
    job_id=0
    for model in "${models[@]}"; do
        for layer in "${layers[@]}"; do
            for task in "${tasks[@]}"; do
                gpu=$((job_id % 4))

                srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:$gpu \
                    python telepathy/linear_probe_baseline.py \
                    --model_id "$model" \
                    --layer_index $layer \
                    --task $task \
                    --train_samples 1000 \
                    --test_samples -1 \
                    --output_dir "$OUTPUT_BASE/phase2/probe_$(basename $model)_L${layer}_${task}" \
                    > "$OUTPUT_BASE/phase2/probe_$(basename $model)_L${layer}_${task}.log" 2>&1 &

                job_id=$((job_id + 1))

                # Batch control: wait every 4 jobs
                if [ $((job_id % 4)) -eq 0 ]; then
                    sleep 5  # Small delay to ensure jobs start
                fi
            done
        done
    done
}

# Phase 3: Fair Baselines (6 hours)
run_phase3_parallel() {
    # Telepathy on GPU 0-1
    for task in sst2 agnews trec; do
        srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:0 \
            python telepathy/train_telepathy_${task}.py \
            --compression_factor 8 \
            --output_dir "$OUTPUT_BASE/phase3/telepathy_${task}" \
            > "$OUTPUT_BASE/phase3/telepathy_${task}.log" 2>&1 &
    done

    # LLMLingua on GPU 2
    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:2 \
        bash -c "
            for task in sst2 agnews trec; do
                python scripts/run_llmlingua_baseline.sh \
                    --task \$task \
                    --compression_rate 0.125 \
                    --output_dir $OUTPUT_BASE/phase3/llmlingua_\$task
            done
        " > "$OUTPUT_BASE/phase3/llmlingua.log" 2>&1 &

    # Direct prompting on GPU 3
    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:3 \
        bash -c "
            for task in sst2 agnews trec; do
                python telepathy/run_llama_baselines.py \
                    --task \$task \
                    --output_dir $OUTPUT_BASE/phase3/direct_\$task
            done
        " > "$OUTPUT_BASE/phase3/direct.log" 2>&1 &
}

# Phase 4: Latency (2 hours)
run_phase4_parallel() {
    local cf_values=(4 8 16 32)
    local bs_values=(1 4 8 16)
    local job_id=0

    for cf in "${cf_values[@]}"; do
        for bs in "${bs_values[@]}"; do
            gpu=$((job_id % 4))

            srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:$gpu \
                python telepathy/benchmark_batched_latency.py \
                --compression_factor $cf \
                --batch_size $bs \
                --num_iterations 100 \
                --output_dir "$OUTPUT_BASE/phase4/latency_cf${cf}_bs${bs}" \
                > "$OUTPUT_BASE/phase4/latency_cf${cf}_bs${bs}.log" 2>&1 &

            job_id=$((job_id + 1))

            # Control parallelism
            if [ $((job_id % 4)) -eq 0 ]; then
                sleep 2
            fi
        done
    done
}

# Phase 5: XSUM Generation (4 hours)
run_phase5_parallel() {
    # Train XSUM bridge on GPU 0-1
    srun --exclusive -N1 -n1 --gpus=2 --gpu-bind=map_gpu:0,1 \
        python telepathy/train_xsum_bridge.py \
        --train_samples 1000 \
        --test_samples 500 \
        --batch_size 2 \
        --output_dir "$OUTPUT_BASE/phase5/xsum_bridge" \
        > "$OUTPUT_BASE/phase5/xsum_bridge.log" 2>&1 &

    # Baseline XSUM on GPU 2-3
    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:2 \
        python telepathy/eval_xsum_bridge.py \
        --model meta-llama/Llama-3.1-1B-Instruct \
        --test_samples 500 \
        --output_dir "$OUTPUT_BASE/phase5/xsum_llama" \
        > "$OUTPUT_BASE/phase5/xsum_llama.log" 2>&1 &

    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:3 \
        python telepathy/eval_xsum_bridge.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --test_samples 500 \
        --output_dir "$OUTPUT_BASE/phase5/xsum_qwen" \
        > "$OUTPUT_BASE/phase5/xsum_qwen.log" 2>&1 &
}

# Phase 6: Model Size Ablations (2 hours)
run_phase6_parallel() {
    # Small models on GPU 0
    srun --exclusive -N1 -n1 --gpus=1 --gpu-bind=map_gpu:0 \
        python telepathy/run_ablation_experiments.py \
        --sender meta-llama/Llama-3.1-1B-Instruct \
        --receiver Qwen/Qwen2.5-1.5B-Instruct \
        --task sst2 \
        --batch_size 4 \
        --output_dir "$OUTPUT_BASE/phase6/ablation_small" \
        > "$OUTPUT_BASE/phase6/ablation_small.log" 2>&1 &

    # Medium models on GPU 1-2
    srun --exclusive -N1 -n1 --gpus=2 --gpu-bind=map_gpu:1,2 \
        python telepathy/run_ablation_experiments.py \
        --sender meta-llama/Meta-Llama-3.1-8B-Instruct \
        --receiver mistralai/Mistral-7B-Instruct-v0.3 \
        --task sst2 \
        --batch_size 1 \
        --output_dir "$OUTPUT_BASE/phase6/ablation_medium" \
        > "$OUTPUT_BASE/phase6/ablation_medium.log" 2>&1 &
}

# Main execution
echo "Starting parallel experiment execution at $(date)"
echo "Output directory: $OUTPUT_BASE"

# Run phases with expected durations
run_phase 1 6  # 6 hours
run_phase 2 4  # 4 hours
run_phase 3 6  # 6 hours
run_phase 4 2  # 2 hours
run_phase 5 4  # 4 hours
run_phase 6 2  # 2 hours

# Total: 24 hours

# Aggregate results
echo "Aggregating results..."
python telepathy/aggregate_results.py --input_dir "$OUTPUT_BASE" --output summary.json

echo "Experiment suite completed at $(date)"
```

## Monitoring and Control

### Real-time Monitoring

```bash
# Check job status
squeue -u $USER

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check specific GPU process
nvidia-smi -i 0 -l 1  # Monitor GPU 0

# View logs in real-time
tail -f runs/parallel_*/phase1/*.log

# Check memory usage per GPU
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Job Control

```bash
# Cancel specific subjob
scancel <job_id>_<task_id>

# Cancel all jobs
scancel -u $USER

# Hold/release jobs
scontrol hold <job_id>
scontrol release <job_id>

# Modify time limit
scontrol update job=<job_id> TimeLimit=30:00:00
```

## Resource Optimization

### Dynamic GPU Assignment

The plan includes dynamic GPU assignment based on experiment requirements:

1. **Memory-intensive tasks** (dual models) → GPUs 0-1
2. **Compute-intensive tasks** (training) → Distributed across all GPUs
3. **I/O-intensive tasks** (evaluation) → GPUs 2-3

### Batch Size Optimization

| Model Size | Dual Model Batch | Single Model Batch |
|------------|------------------|-------------------|
| Small (1-2B) | 4-8 | 16-32 |
| Medium (7-8B) | 1-2 | 4-8 |
| Large (13B+) | 1 | 2-4 |

### Pipeline Optimization

- **Overlapping I/O and compute**: While GPUs 0-1 train, GPUs 2-3 evaluate
- **Staged execution**: Critical experiments first, optional ablations later
- **Checkpoint management**: Save checkpoints to shared storage for resilience

## Failure Recovery

### Checkpoint Strategy

```bash
# Save checkpoints every epoch
--save_strategy epoch
--save_total_limit 2

# Resume from checkpoint
--resume_from_checkpoint runs/checkpoint_dir
```

### Automatic Retry

```bash
# Wrapper script with retry logic
for attempt in 1 2 3; do
    srun command && break
    echo "Attempt $attempt failed, retrying..."
    sleep 60
done
```

## Expected Timeline

| Hour | Phase | GPUs 0-1 | GPUs 2-3 |
|------|-------|----------|----------|
| 0-6 | Phase 1 | Statistical experiments | Baseline preparations |
| 6-10 | Phase 2 | Linear probes (layers 0-15) | Linear probes (layers 16-31) |
| 10-16 | Phase 3 | Telepathy training | LLMLingua + Direct prompting |
| 16-18 | Phase 4 | Latency tests (CF 4,8) | Latency tests (CF 16,32) |
| 18-22 | Phase 5 | XSUM training | XSUM baselines |
| 22-24 | Phase 6 | Small/medium ablations | Large ablations + cleanup |

## Success Metrics

- **Phase completion**: All 6 phases complete within 24 hours
- **GPU utilization**: >80% average utilization across all GPUs
- **Memory efficiency**: No OOM errors with specified batch sizes
- **Result quality**: All experiments produce valid JSON outputs
- **Statistical rigor**: 3 seeds with confidence intervals for key results

## Conclusion

This parallel execution plan maximizes GPU utilization while respecting memory constraints. By carefully orchestrating experiments across 4 H100 GPUs using SLURM's `srun` command, we can complete all required experiments within the 24-hour window. The plan includes dynamic resource allocation, failure recovery mechanisms, and comprehensive monitoring to ensure successful execution.