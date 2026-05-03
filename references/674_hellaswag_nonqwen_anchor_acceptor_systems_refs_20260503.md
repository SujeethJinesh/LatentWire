# HellaSwag Non-Qwen Anchor Acceptor And Systems References

Date: 2026-05-03

## Purpose

This memo records the literature and systems boundary after the cautious
TinyLlama-to-Phi anchor/quantile acceptor gate. The result is not a positive
receiver method, but it strengthens the byte/exposure story by preserving
packet-only accuracy with a `1B` raw / `4B` framed receiver-visible packet.

## Evidence Boundary

- Artifact:
  `results/source_private_hellaswag_nonqwen_anchor_acceptor_gate_20260503_validation1024_2048/`
- Weighted Phi target-only accuracy: `0.263021`.
- Weighted TinyLlama packet-only accuracy: `0.506510`.
- Weighted cautious anchor acceptor accuracy: `0.506510`.
- Weighted target-or-packet oracle accuracy: `0.619792`.
- Selected receiver-visible payload: `1B` raw / `4B` framed.
- Decision: receiver improvement fails, candidate-only preservation passes.

## Novelty Boundary

The defensible claim is not "anchor-relative latent communication." The narrow
claim is:

> A fixed-byte, source-private task-evidence packet protocol whose receiver is
> evaluated under packet-preserving acceptance, matched-byte source-choice
> controls, and byte/exposure comparisons against text, prompt, and KV/cache
> communication.

This is distinct only if the packet contract and destructive controls remain
central.

## Primary Related Work

- Relative Representations:
  https://openreview.net/forum?id=SrC-nwieGJ
  - Provides anchor-relative common coordinates. LatentWire can reuse this idea
    only as a coordinate device, not as the main novelty.
- Product of Invariances:
  https://openreview.net/forum?id=vngVydDWft
  - Builds richer invariant representation spaces for reuse/stitching. Our
    boundary is per-example fixed-byte packet communication.
- Semantic channel equalization:
  https://arxiv.org/abs/2405.13511
  - Pressures any broad "common semantic channel" claim.
- Model stitching:
  https://arxiv.org/abs/2106.07682
  - A compatibility-layer baseline for frozen representation spaces, but it
    does not enforce a source-private fixed-byte packet.
- SVCCA:
  https://arxiv.org/abs/1706.05806
  and CKA:
  https://arxiv.org/abs/1905.00414
  - Representation comparison diagnostics, not packet protocols.
- Prefix-Tuning:
  https://aclanthology.org/2021.acl-long.353/
  and Gist Tokens:
  https://arxiv.org/abs/2304.08467
  - Required prompt-compression/soft-token baselines if we move back to learned
    continuous tokens.
- C2C:
  https://arxiv.org/abs/2510.03215
  and KVComm:
  https://arxiv.org/abs/2510.03346
  - Direct KV/cache communication competitors. They transfer source state; our
    current differentiator is lower-exposure boundary traffic.
- QJL:
  https://arxiv.org/abs/2406.03482
  and TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Quantized vector/KV codec baselines. They motivate byte-floor comparisons
    but are not task-evidence packet protocols.
- KIVI:
  https://arxiv.org/abs/2402.02750
  and KVQuant:
  https://arxiv.org/abs/2401.18079
  - Additional low-bit KV cache baselines for future NVIDIA systems rows.
- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
  and SGLang:
  https://arxiv.org/abs/2312.07104
  - Native serving stacks for later TTFT/TPOT/goodput comparisons.
- FlashAttention:
  https://arxiv.org/abs/2205.14135
  FlashAttention-2:
  https://arxiv.org/abs/2307.08691
  FlashAttention-3:
  https://arxiv.org/abs/2407.08608
  - Kernel-floor references for future GPU serving measurements.

## Reviewer Implication

This result should be reported as a disciplined non-overclaim:

- anchor/quantile acceptors did not beat packet-only;
- one-example select-split gains were rejected because they harmed eval;
- the candidate-only packet preserves the non-Qwen packet row at `1B` raw;
- ICLR acceptance still requires a true positive receiver or a stronger packet
  method on larger strict slices.

## Next Systems Rows

Mac-local rows to keep now:

- raw packet bytes;
- `64B` cache-line rounded bytes;
- batch-packed bytes/request for batch sizes `{1, 8, 16, 32, 64, 128}`;
- source exposure flags;
- accuracy/control gap per byte.

NVIDIA rows later:

- target-only BF16/FP8 serving;
- packet receiver path;
- query-aware text and structured text relay;
- C2C/KV-style source cache transfer or approximation;
- KVComm-style `{10%, 30%, 50%, 100%}` layer sharing;
- vLLM/SGLang KV reuse and quantized KV rows;
- TTFT, TPOT, tok/s, peak HBM, and HBM traffic counters.
