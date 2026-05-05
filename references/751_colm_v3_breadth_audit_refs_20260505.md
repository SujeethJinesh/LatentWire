# COLM v3 Breadth Audit References

Date: 2026-05-05

This memo records the primary sources used when adding the COLM_v3 reviewer-pack
benchmark and latest-model breadth audit. The breadth rows are supporting
diagnostics only; they do not strengthen the main ARC/OpenBookQA claim beyond
the controls already reported in the paper.

## Benchmark Breadth

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Primary source: https://aclanthology.org/P19-1472/
   - Use in COLM_v3: benchmark citation for the bounded validation-1024 packet
     row in `benchmark_breadth.csv`.
   - Boundary: HellaSwag is not promoted to a full-validation headline because
     the terminal-tail strict jackknife gate remains blocked.

2. CommonsenseQA: A Question Answering Challenge Targeting Commonsense
   Knowledge
   - Primary source: https://aclanthology.org/N19-1421/
   - Use in COLM_v3: benchmark citation for the validation diagnostic row.
   - Boundary: the row is target-positive but same-byte text nearly saturates
     it, so it is diagnostic rather than headline evidence.

3. SciQ: Crowdsourcing Multiple Choice Science Questions
   - Primary source: https://arxiv.org/abs/1707.06209
   - Use in COLM_v3: bridge-contract citation for a source-private split/control
     surface.
   - Boundary: SciQ has no positive packet result yet.

## Latest-Model Emitter Breadth

Hugging Face Hub repository metadata was checked for the exact model IDs used in
the reviewer-pack latest-model matrix:

1. Qwen/Qwen3.5-0.8B
   - Model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
   - Local role: CPU n160 source-packet emitter smoke.

2. Qwen/Qwen3.5-2B
   - Model card: https://huggingface.co/Qwen/Qwen3.5-2B
   - Local role: CPU n160 source-packet emitter smoke.

3. Qwen/Qwen3.5-4B
   - Model card: https://huggingface.co/Qwen/Qwen3.5-4B
   - Local role: CPU n64 source-packet emitter smoke.

4. google/gemma-4-E2B-it
   - Model card: https://huggingface.co/google/gemma-4-E2B-it
   - Local role: MPS/CPU source-packet emitter smoke, including n500 MPS.

5. ibm-granite/granite-3.3-2b-instruct
   - Model card: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct
   - Local role: CPU trace-no-hint source-packet emitter smoke; raw-log/no-trace
     collapse kept as a negative prompt-contract boundary.

6. allenai/OLMo-2-0425-1B-Instruct
   - Model card: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
   - Local role: negative or unverified row; do not use as a positive breadth
     claim.

Boundary for all latest-model rows: these are source-packet emitter smokes on a
synthetic hidden-repair benchmark. They are useful for reviewer visibility into
prompt/protocol portability, but they are not ARC/OpenBookQA benchmark evidence
and should not be cited as the main method result.
