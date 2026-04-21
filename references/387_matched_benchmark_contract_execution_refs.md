# 387. Matched Benchmark Contract Execution References

Date: 2026-04-21

This memo freezes the benchmark contract implied by the local competitor repos.
The main conclusion is simple: there is no single fair mixed table across
`C2C`, `KVComm`, `LatentMAS`, `lm-eval`, and `OpenCompass`. The paper should
use two main suites plus one appendix suite.

## Local repo read

1. C2C
   - Repo: `references/repos/C2C`
   - Key files:
     - `script/evaluation/unified_evaluator.py`
     - `recipe/eval_recipe/unified_eval.yaml`
   - Read: C2C is a real cross-model communication comparator on reasoning and
     knowledge tasks such as `gsm8k`, `gpqa_diamond`, `ARC-Challenge`,
     `mmlu-redux`, and `longbench`.

2. KVComm
   - Repo: `references/repos/KVComm`
   - Key files:
     - `dataloader/__init__.py`
     - `dataloader/base_evaluator.py`
     - `com.py`
   - Read: KVComm's clean overlap is LongBench-style long-context QA,
     especially `hotpotqa`, `qasper`, `multifieldqa_en`, `twowikimqa`, and
     `musique`. Its repo-native scorer is not the same thing as a shared main
     table scorer.

3. LatentMAS
   - Repo: `references/repos/LatentMAS`
   - Key file:
     - `run.py`
   - Read: LatentMAS is a same-backbone multi-agent system, not a clean
     heterogeneous sender/receiver cross-model communication baseline. It
     belongs in an appendix or same-model table unless we explicitly port it.

4. lm-evaluation-harness
   - Repo: `references/repos/lm-evaluation-harness`
   - Key file:
     - `lm_eval/tasks/gsm8k/gsm8k.yaml`
   - Read: Use `lm-eval` as a scoring and orchestration harness, not as a
     competitor method. Its repo-native prompts and shot counts are not a
     strict head-to-head contract by default.

5. OpenCompass
   - Repo: `references/repos/opencompass`
   - Key file:
     - `opencompass/configs/datasets/gsm8k/gsm8k_gen_1d7fe4.py`
   - Read: Same issue as `lm-eval`: useful harness, not a method row. Its
     default GSM8K prompt contract differs from `lm-eval`.

## Frozen paper contract

### Main Table A: cross-model reasoning

- Tasks:
  - `gsm8k`
  - `gpqa_diamond`
  - `arc_challenge`
- Primary metrics:
  - GSM8K normalized numeric EM
  - GPQA Diamond accuracy
  - ARC-Challenge accuracy
- Rows:
  - `source_only`
  - `receiver_only`
  - `text_exchange`
  - `ours`
  - `c2c_generate`
  - `c2c_two_stage_generate`
- Exclusions:
  - do not place `KVComm` here
  - do not place `LatentMAS` here

### Main Table B: cross-model long-context QA

- Tasks:
  - `hotpotqa`
  - `qasper`
  - `multifieldqa_en`
  - `2wikimqa`
  - `musique`
- Primary metric:
  - official LongBench `qa_f1`
- Rows:
  - `source_only`
  - `receiver_only`
  - `text_exchange`
  - `ours`
  - `kvcomm`
  - `kvcomm_nld`
  - `kvcomm_cipher`
  - `c2c_generate`
- Exclusions:
  - do not use KVComm's repo-native boolean `f1_match` as the main paper metric

### Appendix Table C: same-backbone / code

- Tasks:
  - `humaneval_plus`
  - `mbpp_plus`
  - optionally `medqa`, `aime2024`, `aime2025`
- Primary metrics:
  - `pass@1` for code
  - accuracy for the rest
- Rows:
  - `receiver_only`
  - `text_mas`
  - `latent_mas`
  - `ours_same_model`
- Exclusions:
  - do not mix `C2C` or `KVComm` into this table

## Normalization rules

1. Freeze one sender/receiver pair per main suite.
2. Use greedy decoding everywhere in the main tables:
   - `do_sample=false`
   - `temperature=0.0`
   - `num_return_sequences=1`
3. Force C2C into `answer_method=generate` for main-table comparisons.
4. Use one frozen external prompt per task family.
5. Main-table prompts should be zero-shot and no-CoT.
6. Score all saved predictions offline with one shared scorer per task family.
7. For LongBench overlap, rescore every method with one official LongBench
   scorer and map `twowikimqa -> 2wikimqa`.
8. Fix long-context output caps:
   - `hotpotqa=48`
   - `qasper=128`
   - `multifieldqa_en=64`
   - `2wikimqa=32`
   - `musique=48`
9. Keep calibration disjoint from scored examples for KVComm and any learned
   LatentWire remap/router.
10. Use HF backend for any LatentMAS appendix row.

## Minimum smoke tests before any headline row

1. Sample alignment
   - identical example IDs on a 32-example slice for every method row

2. Parser coverage
   - 100% choice extraction on GPQA and ARC
   - no empty numeric extraction on GSM8K
   - no silent drops

3. Determinism
   - rerun the same 32-example slice twice
   - greedy rows must match exactly

4. Calibration hygiene
   - prove calibration indices and scored indices are disjoint

5. Harness parity
   - `receiver_only` through native script and shared scorer must agree on the
     same 32-example slice

6. Context-budget parity
   - log truncation rate and reject rows with hidden context mismatch

7. Code-eval sanity
   - EvalPlus / code scorers must pass a 10-example smoke run before any
     `pass@1` claim

8. Method-load sanity
   - C2C checkpoint load
   - KVComm communication row
   - LatentMAS HF row
   - our method row
   all need one tiny run without fallback or parser failure

## Practical implication

- The benchmark track is now a frozen execution contract, not a moving target.
- If we keep a single mixed table, the comparison will be technically unsound.
- The method-discovery track should continue separately until `ours` is strong
  enough to deserve the full contract.
