# References: HellaSwag Public Receiver Repair Probe

Date: 2026-05-01

## Local Artifact

- `paper/source_private_hellaswag_public_receiver_repair_probe_20260501.md`
- `results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024/`

## Primary References

- HellaSwag: Can a Machine Really Finish Your Sentence?
  https://arxiv.org/abs/1905.07830
  Role: adversarial continuation benchmark and the current hard non-science
  diagnostic surface.

- Calibrate Before Use: Improving Few-Shot Performance of Language Models.
  https://arxiv.org/abs/2102.09690
  Role: motivates answer-prior and prompt-bias calibration before claiming MCQ
  gains from a communication packet.

- Large Language Models Are Not Robust Multiple Choice Selectors.
  https://arxiv.org/abs/2309.03882
  Role: motivates option-order and option-ID controls; relevant to the
  source-label-copy threat.

- Selective Classification for Deep Neural Networks.
  https://arxiv.org/abs/1705.08500
  Role: frames the repair problem as deciding when to trust, reject, or defer
  from a source prediction.

- Energy-based Out-of-distribution Detection.
  https://arxiv.org/abs/2010.03759
  Role: supports energy/confidence features as a future source-error signal,
  beyond top-1/top-2 margin.

- Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
  Methods.
  https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/
  Role: cheap rank-fusion baseline for source rank, receiver rank, and
  source-score calibration.

## Systems / Related-Work Boundary

- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  https://arxiv.org/abs/2510.03215
  Boundary: C2C transfers projected/fused KV cache state, not a few-byte
  source-private task packet.

- KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
  Multi-agent Systems.
  https://arxiv.org/abs/2510.12872
  Boundary: KV-cache reuse/communication is the native systems comparator, not
  the same byte-rate or threat model.

- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
  https://openreview.net/forum?id=F7rUng23nw
  Boundary: selective KV sharing should be compared in NVIDIA/vLLM rows once
  hardware is available.

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  https://arxiv.org/abs/2504.19874
  Boundary: relevant inspiration for random rotations and residual sketches,
  but it quantizes vectors/KV state rather than train-only MCQ repair packets.

- Efficient Memory Management for Large Language Model Serving with
  PagedAttention.
  https://arxiv.org/abs/2309.06180
  Boundary: serving-system baseline for future native throughput, TTFT, and
  memory rows.

## Claim Boundary

This result is a falsification artifact. It shows that a public train-only
lexical receiver plus source top-2 hint does not beat visible source-label
copy on frozen HellaSwag first-1024. The next live method must learn a
source-error signal from train-split source scores or source hidden summaries.
