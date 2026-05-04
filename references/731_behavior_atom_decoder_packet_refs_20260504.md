# Behavior-Atom Decoder Packet References

Date: 2026-05-04

## Why This Branch Exists

The hidden-atom PCA scout showed a small target lift but failed atom knockout
and zero-source controls. The behavior-atom branch therefore fits packet atoms
toward target residual behavior rather than source reconstruction variance.
This keeps Sparse Resonance Packets distinct from dense activation/KV transfer:
the packet is a few quantized atom IDs and coefficients, not source hidden
vectors, KV cache blocks, or natural-language text.

## Dense Communication Competitors

- [Cache-to-Cache: Direct Semantic Communication Between Large Language
  Models](https://arxiv.org/abs/2510.03215) is the main dense competitor.
  C2C learns to project and fuse the source model's KV cache into the target
  cache and reports accuracy and latency gains over text communication. SRP
  should cite C2C as the high-bandwidth positive baseline, then compete on
  source exposure and utility per byte.
- [KVComm: Enabling Efficient LLM Communication through Selective KV
  Sharing](https://arxiv.org/abs/2510.03346) selectively shares informative KV
  pairs and reports comparable performance while transmitting a fraction of
  layers' KV pairs. This is still internal-state transfer; SRP differs by
  sending fixed-format sparse packets with no source KV/hidden exposure.
- [Communicating Activations Between Language Model
  Agents](https://arxiv.org/abs/2501.14082) communicates intermediate
  activations between LM agents. It is an activation-language competitor, not a
  source-private packet protocol.

## Sparse Basis And Behavior-Atom Priors

- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410) motivates
  variable per-example sparse activation budgets. If linear behavior atoms keep
  failing, BatchTopK is the natural Mac-local next step for learned sparse
  packet atoms.
- [Quantifying Feature Space Universality Across Large Language Models via
  Sparse Autoencoders](https://arxiv.org/abs/2410.06981) supports the common
  feature-basis hypothesis, but it is representational evidence rather than a
  communication protocol. SRP still needs causal packet controls.
- [Sparse Crosscoders for Cross-Layer Features and Model
  Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html)
  motivates shared/private feature partitions. SRP can differ by using
  crosscoder atoms as a rate-limited message, evaluated by downstream utility.
- [Transcoders Find Interpretable LLM Feature
  Circuits](https://huggingface.co/papers/2406.11944) and [Transcoders Beat
  Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2501.18823)
  motivate behavior-oriented sparse features. This supports training SRP atoms
  against target residual behavior instead of reconstruction alone.

## Quantization And Systems Boundary

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion
  Rate](https://huggingface.co/papers/2504.19874) is an important low-bit KV
  compression comparator. Its reported KV quality neutrality around 3.5 bits
  per channel means SRP should not claim a generic cache-compression win.
- [KVQuant](https://arxiv.org/abs/2401.18079) is another strong KV byte-floor
  baseline with sub-4-bit KV quantization and custom kernels. SRP should report
  byte movement as a different utility-per-byte point, not as measured GPU
  throughput.

## Controls This Branch Must Survive

The subagent benchmark review reinforced that same-source-choice wrong-row is
mandatory: if a packet from another row with the same source-selected answer
works as well as matched, the method is source-choice transfer rather than
row-specific latent communication.

The current behavior-atom scout fails this control, plus top-atom knockout and
Qwen-substitution. The next receiver should explicitly isolate source-specific
innovation by subtracting the zero-packet residual and selecting gates based on
matched-vs-null packet gain.

## Takeaway

The literature supports sparse, interpretable, cross-model feature bases, but
it does not already solve LatentWire's target problem: fixed-byte,
source-private, quantized packets that improve a receiver under destructive
controls. The novelty path remains viable only if the next branch proves that
the sparse packet carries row-specific source evidence beyond target-cache and
source-choice shortcuts.
