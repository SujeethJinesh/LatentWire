# HellaSwag Hidden-Innovation Stability References

## Local Result

- Artifact: `results/source_private_hellaswag_hidden_innovation_stability_gate_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_stability_gate.json`
- Paper memo: `paper/source_private_hellaswag_hidden_innovation_stability_gate_20260501.md`
- Gate result: anchored `score_hidden_residual` passes `5/5` cached train/dev split seeds on HellaSwag validation1024.
- Main margin: mean/min delta vs best label-copy is `+0.040820/+0.032227`; min paired CI95 low is `+0.006836`.
- Diagnostic: unrestricted model selection passes only `3/5` split seeds and can drift into `score_only` or `hidden_residual_only` shortcuts.
- Packet contract: `2B` raw, `5B` framed; no source text, KV, raw hidden vectors, or raw scores cross the boundary.

## Primary Sources Checked

- HellaSwag: `https://arxiv.org/abs/1905.07830`.
- Sparse Autoencoders Find Highly Interpretable Features in Language Models: `https://arxiv.org/abs/2309.08600`.
- Scaling and evaluating sparse autoencoders: `https://arxiv.org/abs/2406.04093`.
- Towards Monosemanticity: `https://transformer-circuits.pub/2023/monosemantic-features/`.
- Scaling Monosemanticity: `https://transformer-circuits.pub/2024/scaling-monosemanticity/`.
- Relative Representations Enable Zero-Shot Latent Space Communication: `https://arxiv.org/abs/2209.15430`.
- Communicating Activations Between Language Model Agents: `https://arxiv.org/abs/2501.14082`.
- Representation Engineering: `https://arxiv.org/abs/2310.01405`.
- Inference-Time Intervention: `https://arxiv.org/abs/2306.03341`.
- Cache-to-Cache communication: `https://arxiv.org/abs/2510.03215`.
- KVComm: `https://openreview.net/forum?id=F7rUng23nw`.
- QJL: `https://arxiv.org/abs/2406.03482`.
- TurboQuant: `https://arxiv.org/abs/2504.19874`.
- Diffusion-LM: `https://arxiv.org/abs/2205.14217`.
- Diffusion Transformers: `https://arxiv.org/abs/2212.09748`.
- LaDiR: `https://arxiv.org/abs/2510.04573`.

## Uniqueness Boundary

The stabilized claim is not that LatentWire invents sparse features, activation steering, model stitching, representation alignment, KV-cache communication, vector quantization, or diffusion denoising. Those are prior areas with direct related work.

The defensible novelty is narrower: the source uses score-plus-hidden-residual evidence locally to choose a tiny source-private candidate/confidence record. The receiver sees only the fixed-byte record and public candidate side information, yet the packet beats source-label copy, trained label-copy, zero-hidden, wrong-example-hidden, and candidate-roll controls across cached split seeds.

The unrestricted selector result is a useful reviewer defense. It shows that allowing arbitrary hidden or score-only views is unstable, so the promoted method must be the anchored score-plus-hidden-residual rule.

## Next Citation-Backed Branch

If the next HellaSwag gate fails on new train-row samples or full validation, the highest-value branch is an SAE/common-basis packet: train a sparse dictionary over candidate hidden innovations, send top-k feature IDs/signs as the packet, and test feature-ID shuffle, feature-value shuffle, wrong-example sparse code, and label-shuffle controls. This is inspired by SAE and relative-representation work, but the claim must remain source-private byte-bounded repair rather than general latent language.
