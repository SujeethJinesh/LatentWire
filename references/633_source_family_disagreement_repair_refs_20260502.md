# Source-Family Disagreement Repair References, 2026-05-02

## Gate

LatentWire is not ICLR-ready. The live blocker is whether fixed-byte
source-private communication can avoid harm when source families disagree with
the target. The useful next branch is therefore not a wider benchmark, but a
Mac-feasible confidence/abstention/source-selection gate with paired
uncertainty and destructive controls.

## Local Evidence Read

- `paper/reviewer_feedback.md`: current tiny-slice deltas are not reliable;
  evaluation needs larger frozen slices, seed repeats, paired uncertainty, and
  oracle/headroom diagnostics.
- `results/asym_kv_qwen_20260421/telemetry_summary_20260421.md`: random/source
  routing is unstable; route perturbations can rescue and harm examples.
- `paper/source_private_memory_traffic_ledger_20260430.md`: byte-boundary and
  source-private packet evidence is useful, but not a substitute for a learned
  receiver/selector.
- `paper/source_private_native_readiness_ledger_20260501.md`: native
  throughput/systems superiority over KV/cache baselines remains unsafe.

## Primary Sources

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks,"
  NeurIPS 2017. https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- Geifman and El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated
  Reject Option," ICML 2019. https://proceedings.mlr.press/v97/geifman19a.html
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017.
  https://arxiv.org/abs/1706.04599
- Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks," 2016.
  https://arxiv.org/abs/1610.02136
- Angelopoulos et al., "Conformal Risk Control," 2022.
  https://arxiv.org/abs/2208.02814
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer," 2017. https://arxiv.org/abs/1701.06538
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with
  Simple and Efficient Sparsity," 2021. https://arxiv.org/abs/2101.03961
- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources," IEEE
  TIT 1973. DOI: 10.1109/TIT.1973.1055037.
- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder," IEEE TIT 1976. Public PDF:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- Pradhan and Ramchandran, "Distributed Source Coding Using Syndromes
  (DISCUS): Design and Construction," IEEE TIT 2003. DOI:
  10.1109/TIT.2002.808103.
- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," 2025. https://arxiv.org/abs/2510.03215
- "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing,"
  2025. https://arxiv.org/abs/2510.03346
- "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
  Multi-agent Systems," 2025. https://arxiv.org/abs/2510.12872
- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," 2025. https://arxiv.org/abs/2504.19874
- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation,"
  ACL 2021. https://aclanthology.org/2021.acl-long.353/

## Positioning

Frame LatentWire as fixed-byte, source-private, decision-gated communication:
send a tiny source-derived evidence packet only when calibrated receiver-side
risk predicts positive conditional value over target-alone. Do not claim
general KV-cache fusion superiority or native serving superiority. C2C and
KVComm/KVCOMM are full/partial KV communication and reuse baselines; TurboQuant
is a compression baseline for same-model KV/cache vectors; prefix tuning is a
continuous prompt adaptation baseline, not source-conditioned fixed-byte
communication.

## Recommended Mac Gate

Run a paired selective source gate over the existing frozen ARC/OpenBookQA
packet rows:

1. For each item, log target-alone margin/entropy, packet receiver margin,
   source-family identity, source-target disagreement flag, packet id, and
   correctness for target-alone/packet/destructive controls.
2. Calibrate a scalar accept score on train only:
   `score = packet_margin - target_margin - lambda * source_disagreement`.
   Sweep thresholds to produce risk-coverage and delta-coverage curves.
3. Evaluate held-out same-family and strict cross-family pairs with paired
   bootstrap/McNemar, seed repeats, and oracle headroom.
4. Pass only if accepted examples improve over target-alone, abstained examples
   are neutral, destructive controls stay within target +0.03, and
   cross-family disagreement does not erase the gain.

Decision: this is cheaper and more diagnostic than a new connector branch. It
directly tests whether the current failure is repairable by selective
communication or whether source-family disagreement kills the fixed-byte packet
story.
