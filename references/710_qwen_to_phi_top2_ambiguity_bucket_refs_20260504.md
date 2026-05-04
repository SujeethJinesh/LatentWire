# Reference Memo 710: Qwen-To-Phi Top1/Top2 Ambiguity Bucket Gate

Date: 2026-05-04

## Local Result Boundary

The official-train Qwen-to-Phi top1/top2 ambiguity-bucket gate failed. The
receiver-visible packet exposed only source top1/top2 IDs plus quantized
decision-syndrome bins, and the receiver used Phi-local score bins as side
information. The official-train selector chose `no_op`, giving `0.467448`
accuracy, exactly tied with fixed hybrid and no-syndrome top-pair controls.
The source top1/top2 oracle remains high at `0.675781`, so the headroom exists
but is not captured by score-level ambiguity buckets.

## Closest Prior Work And Boundary

- Knowledge distillation / dark knowledge: Hinton et al. motivate that a
  teacher's score distribution contains information beyond hard labels.
  LatentWire's top-pair packet is inference-time, fixed-byte, and
  source-private rather than training-time full-distribution distillation.
  Source: https://arxiv.org/abs/1503.02531

- Prefix/prompt tuning: learned soft prompts and prefixes condition frozen
  language models. LatentWire must remain instance-specific and source-derived,
  not a persistent learned prompt with no source evidence.
  Sources: https://arxiv.org/abs/2101.00190,
  https://arxiv.org/abs/2104.08691

- Context compression / gist tokens / LLMLingua: these methods compress text
  contexts or prompts. LatentWire's defensible boundary is not text
  compression; it is source-private decision evidence.
  Sources: https://arxiv.org/abs/2304.08467,
  https://arxiv.org/abs/2310.05736

- Relative representations, SAEs, and crosscoders: these occupy broad
  representation-alignment and common-feature territory. A shared basis is not
  enough for novelty; it must causally improve a source-private receiver under
  destructive controls.
  Sources: https://arxiv.org/abs/2209.15430,
  https://arxiv.org/abs/2309.08600,
  https://transformer-circuits.pub/2024/crosscoders/index.html

- Logit fusion / proxy tuning / contrastive decoding: this is the closest
  method family because it combines source and target score evidence. LatentWire
  must beat equal-byte quantized logit/top-k score packets, not only raw
  full-score controls.
  Sources: https://arxiv.org/abs/2105.03023,
  https://arxiv.org/abs/2210.15097,
  https://arxiv.org/abs/2401.08565

- Selective prediction / learning to defer: receiver calibration overlaps with
  abstain/defer methods. LatentWire should claim fixed-coverage improvement,
  not selective routing or abstention.
  Sources: https://arxiv.org/abs/1705.08500,
  https://arxiv.org/abs/1901.09192

- C2C / KV communication: Cache-to-Cache directly projects and fuses source KV
  cache into a target model and reports higher accuracy and lower latency than
  text communication. This is a mandatory competitor, but it exposes dense KV
  state rather than a fixed-byte source-private packet.
  Source: https://arxiv.org/abs/2510.03215

- KV quantization and vector quantization: QJL, KIVI, KVQuant, and TurboQuant
  define hard byte/quality baselines for compressing dense KV or vector state.
  They are not decision-packet communication protocols, but they are necessary
  byte-floor comparators.
  Sources: https://arxiv.org/abs/2406.03482,
  https://arxiv.org/abs/2402.02750,
  https://arxiv.org/abs/2401.18079,
  https://arxiv.org/abs/2504.19874

- Serving systems: vLLM/PagedAttention and SGLang/RadixAttention define the
  native serving substrates for future GPU claims. Current Mac-local packet
  evidence must not be sold as native HBM/goodput superiority.
  Sources: https://arxiv.org/abs/2309.06180,
  https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html

## Consequence For The Next Gate

This result weakens top-pair score-bucket methods. The next bounded gate should
be an equal-byte quantized source-score comparator inspired by QJL/TurboQuant:
random/sign residual or compact top-k score packets at `1B/2B/4B/8B`, compared
against fixed hybrid, source-index/rank, no-source, target-derived, and
candidate-roll controls. If that fails, score-level packets should be treated
as saturated on Qwen-to-Phi and the project should pivot back to target-native
latent receiver objectives.
