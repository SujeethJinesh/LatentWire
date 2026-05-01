# HellaSwag Hidden-Innovation Train-Sample Stress References

## Local Result

- Artifact: `results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_train_sample_stress.json`
- Stable-ridge diagnostic: `results/source_private_hellaswag_hidden_innovation_train_sample_stress_ridge_stable_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_train_sample_stress.json`
- Paper memo: `paper/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501.md`
- Result: the original train sample seed `1729` passes, but the fresh train sample seed `2027` fails one of three split rows.
- Main default gate: `5/6` split rows pass; train sample pass map is `{'1729': True, '2027': False}`; mean/min delta vs best label-copy is `+0.021484/-0.044922`.
- Stable-ridge diagnostic: constraining ridge choices to `{1000, 10000, 100000}` still fails `5/6`, with mean/min delta `+0.026855/-0.012695`.
- Interpretation: cached split stability was not enough. HellaSwag is weakened as an ICLR headline benchmark until a new train-row sample or full-validation stress clears.

## Primary Sources Checked

- HellaSwag: `https://arxiv.org/abs/1905.07830`.
- Cache-to-Cache, ICLR 2026 poster: `https://openreview.net/forum?id=LeatkxrBCi`.
- KVCOMM: `https://arxiv.org/abs/2510.12872`.
- KVComm OpenReview entry: `https://openreview.net/forum?id=F7rUng23nw`.
- Q-KVComm: `https://arxiv.org/abs/2512.17914`.
- TurboQuant: `https://arxiv.org/abs/2504.19874`.
- TurboESM / TurboQuant plus QJL correction: `https://arxiv.org/abs/2603.26110`.
- Sequential KV cache compression beyond per-vector TurboQuant: `https://arxiv.org/abs/2604.15356`.
- QJL: `https://arxiv.org/abs/2406.03482`.
- Sparse Autoencoders Find Highly Interpretable Features in Language Models: `https://arxiv.org/abs/2309.08600`.
- Scaling and evaluating sparse autoencoders: `https://arxiv.org/abs/2406.04093`.
- Relative Representations Enable Zero-Shot Latent Space Communication: `https://arxiv.org/abs/2209.15430`.
- Communicating Activations Between Language Model Agents: `https://arxiv.org/abs/2501.14082`.
- Diffusion-LM: `https://arxiv.org/abs/2205.14217`.
- Diffusion Transformers: `https://arxiv.org/abs/2212.09748`.
- LaDiR: `https://arxiv.org/abs/2510.04573`.

## Reviewer Boundary

Do not present the dense hidden-innovation HellaSwag row as a stable headline result. It is positive on the original cached train-hidden sample and stable across cached train/dev splits, but it fails the first fresh train-row sample stress.

The result is still useful: it tells us that hidden residuals contain signal, but the sender fit is sample-sensitive. The next scientifically defensible branch is not more threshold tuning; it is either multi-sample aggregation, stronger stability-selected regularization, or a sparse/common-basis packet that makes the source innovation less dependent on dense coordinates.

The uniqueness claim remains source-private and byte-bounded. C2C/KVCOMM/KVComm transmit or reuse high-dimensional KV/cache state; QJL/TurboQuant compress vectors/cache; SAE/relative-representation work builds common feature or anchor bases; diffusion/latent-reasoning work denoises latent trajectories. LatentWire's current claim is a fixed-byte candidate/confidence repair packet with strict no-text/no-KV/no-raw-hidden/no-raw-score exposure. The fresh train-sample failure means we cannot yet claim a robust HellaSwag method.
