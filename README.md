# Telepathy: Continuous Latent Communication for Cross-Model Inference

**MLSys 2025**

A neural bridge enabling cross-model inference through continuous latent tokens, achieving 8x latency reduction over text-based communication.

## Key Results

| Dataset | Telepathy | Text-Relay | Speedup |
|---------|-----------|------------|---------|
| SST-2   | 93.7%     | 41.3%      | 4.7x    |
| AG News | 90.7%     | 1.0%       | 4.7x    |

## Quick Start

```bash
pip install -r requirements.txt

# Train bridge on SST-2
python telepathy/train_telepathy.py --dataset sst2

# Evaluate
python telepathy/eval_telepathy.py --dataset sst2 --checkpoint runs/sst2/bridge.pt

# Reproduce all paper results
sbatch telepathy/submit_enhanced_paper_eval.slurm
```

See [REPRODUCTION.md](REPRODUCTION.md) for detailed reproduction instructions.

## Repository Structure

```
LatentWire/
├── latentwire/          # Core library
│   ├── train.py         # Training loop
│   ├── eval.py          # Evaluation
│   ├── models.py        # Encoder, Adapter, LMWrapper
│   └── ...
├── telepathy/           # Paper experiments
│   ├── train_telepathy.py      # Unified training
│   ├── eval_telepathy.py       # Unified evaluation
│   ├── run_baselines.py        # All baselines
│   ├── run_benchmarks.py       # Latency/throughput
│   ├── run_enhanced_paper_evaluation.py  # Full paper eval
│   └── paper_writing/          # LaTeX source
├── scripts/             # Analysis scripts
└── REPRODUCTION.md      # Detailed reproduction guide
```

## Citation

```bibtex
@inproceedings{telepathy2025,
  title={Telepathy: Continuous Latent Communication for Cross-Model Inference},
  author={...},
  booktitle={MLSys},
  year={2025}
}
```

## License

MIT
