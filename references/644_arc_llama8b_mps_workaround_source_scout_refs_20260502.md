# References: ARC Llama-8B MPS Workaround And Source Scout

Web/literature check: 2026-05-02. Scope: the true non-Qwen
Llama-8B source-family scout and the local Apple MPS attention workaround used
for `results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/`.

## Local Artifact

- Script: `scripts/build_source_private_arc_challenge_llama8b_disagreement_source_scout.py`
- Shared scorer: `scripts/run_source_private_arc_challenge_fixed_packet_gate.py`
- Result: `results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/`
- Source family: `llama3.1_8b_instruct`
- Surface: `144` validation and `473` frozen test TinyLlama-vs-Qwen ARC disagreement rows
- MPS workaround: `attn_implementation=eager`, `choice_batch_size=1`
- Pass gate: `False`
- Test matched/Qwen-substituted/cached-Tiny mean:
  `0.368288/0.317125/0.269345`
- Test CI95 lower bound versus Qwen-substituted: `-0.034937`
- Validation matched/Qwen-substituted mean: `0.355556/0.388889`

## Primary Sources And Boundaries

1. Meta Llama 3.1 model card
   - https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
   - Boundary: Llama 3.1 is a true non-Qwen source family, and the model card
     documents grouped-query attention and ARC-Challenge reporting. A passing
     row here would be meaningful source-family evidence, unlike same-family
     Qwen-1.5B.

2. PyTorch scaled dot product attention and GQA notes
   - https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html
   - Boundary: GQA support is documented as experimental and CUDA-oriented.
     This supports treating the original Apple MPS `mps.matmul` failure as a
     backend execution issue, not a scientific negative.

3. Hugging Face Transformers attention implementations
   - https://huggingface.co/docs/transformers/model_doc/auto
   - https://huggingface.co/docs/transformers/main/attention_interface
   - https://huggingface.co/docs/transformers/main/perf_infer_gpu_one
   - Boundary: `attn_implementation="eager"` is a valid way to avoid automatic
     SDPA routing. LatentWire should log this as a local execution workaround,
     not a new algorithmic contribution.

4. PyTorch MPS environment variables
   - https://docs.pytorch.org/docs/2.11/mps_environment_variables.html
   - Boundary: `PYTORCH_ENABLE_MPS_FALLBACK=1` and
     `PYTORCH_MPS_PREFER_METAL=1` are local backend workarounds. They are not
     paper evidence for native systems speed or memory wins.

5. PyTorch MPS SDPA failure reports
   - https://github.com/pytorch/pytorch/issues/149261
   - Boundary: large SDPA/MPS failures are a known backend class. The
     reviewer-facing result is the completed full scout after workaround, not
     the original crash.

6. Cache-to-cache and KV communication competitors
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://openreview.net/forum?id=F7rUng23nw
   - Boundary: the Llama scout still sends only the selected-answer-derived
     12-byte packet, not source KV/cache state. It therefore remains a
     source-private packet test, but it is not a systems competitor result.

## Claim Policy

Allowed now:

- the Apple MPS hardware blocker is cleared for the local Llama scout;
- the full frozen-disagreement Llama source row is a strict gate failure;
- the source-private contract remains intact: no raw hidden states, text, or
  KV cache cross the boundary;
- Llama-8B source choices show better source-side accuracy than TinyLlama on
  this surface, but the packet result is not robust enough to claim.

Blocked:

- using Llama-8B as the ICLR cross-family positive method;
- claiming source-family generalization from the positive frozen-test mean
  alone;
- claiming native GPU systems wins from a Mac MPS workaround.

Revival condition:

- a distinct Llama prompt/scoring/calibration branch would need validation
  selection, paired uncertainty, and the same Qwen-substituted/cached/source-
  destroying controls before being considered again.
