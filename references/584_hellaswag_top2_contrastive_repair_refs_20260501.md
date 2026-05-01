# HellaSwag Top-2 Contrastive Repair References

Web check: 2026-05-01. Scope: primary sources for the HellaSwag
top-2 contrastive source-error packet, common-basis motivation, MCQ
label-copy controls, and systems comparison boundaries.

## Local Result

Artifact:
`results/source_private_hellaswag_top2_contrastive_repair_probe_20260501_qwen05_train512_validation1024/`

- pass gate: `False`
- selected view: `hidden_score_contrast`
- selected eval accuracy: `0.449219`
- source-label-copy eval accuracy: `0.461914`
- trained label-bias copy eval accuracy: `0.458984`
- selected minus best label-copy: `-0.012695`
- selected minus best zero-hidden control: `+0.006836`
- wrong-example packet accuracy: `0.388672`
- pair-swap control accuracy: `0.266602`
- source top-2 oracle accuracy: `0.715820`

## Method Motivation And Boundary

- Zellers et al., HellaSwag: Can a Machine Really Finish Your Sentence?,
  ACL 2019. HellaSwag is the adversarial continuation benchmark used for this
  frozen validation repair surface. https://arxiv.org/abs/1905.07830
- Zhao et al., Calibrate Before Use, ICML 2021. Contextual calibration
  motivates answer-bias and label-copy controls for few-shot/choice tasks.
  https://arxiv.org/abs/2102.09690
- Zheng et al., Large Language Models Are Not Robust Multiple Choice
  Selectors, ICLR 2024. MCQ option-ID sensitivity motivates treating raw
  source labels and trained label-bias copy as hard baselines.
  https://arxiv.org/abs/2309.03882
- Moschella et al., Relative Representations Enable Zero-Shot Latent Space
  Communication, 2022. Relative/common coordinates motivate the shared-basis
  framing, but the local probe does not yet show positive latent transfer.
  https://arxiv.org/abs/2209.15430
- Lenc & Vedaldi, Understanding Image Representations by Measuring Their
  Equivariance and Equivalence, 2015. Model-stitching/equivalence ideas
  motivate alignment controls for hidden representations.
  https://arxiv.org/abs/1411.5908

## Systems And Quantization Neighbors

- Fu et al., Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models, ICLR 2026. C2C is a high-rate source-KV communication
  baseline with a different state-exposure model. https://arxiv.org/abs/2510.03215
- Shi et al., KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing, ICLR 2026. KVComm motivates native KV-sharing baselines that
  must be compared on GPU serving metrics, not Mac byte accounting alone.
  https://arxiv.org/abs/2510.03346
- Zandieh et al., QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
  with Zero Overhead, 2024. QJL motivates sign-sketch/common-projection
  comparisons for future vector packets. https://arxiv.org/abs/2406.03482
- Zandieh et al., TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate, 2025. TurboQuant motivates residual vector quantization
  branches and byte-floor comparisons. https://arxiv.org/abs/2504.19874

## Safe Boundary

This probe is a killed HellaSwag method branch, not a positive result. It
shows that a top-2 contrastive hidden/score switch signal can beat zero-hidden
controls slightly, but it still fails source-label-copy and trained label-bias
copy. The next live HellaSwag branch must change the representation learner
or denoising objective rather than retuning this last-layer ridge switcher.
