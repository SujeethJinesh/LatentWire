# RotAlign-KV Progress Report

## Plain-English Goal

We are testing whether one language model can hand off its **internal working memory**
to another model, instead of explaining itself in plain text.

The simplest way to think about it:
- `text-to-text` = one model writes notes for the other model
- `RotAlign-KV` = one model tries to pass over part of its "brain state" directly

If this works, the second model should solve reasoning problems better, while sending
less information than a full text explanation.

## What We Tested

### 1. Code reliability tests

We first made sure the machinery itself was not broken:
- unit tests for cache handling, scoring, prompt formatting, calibration, and runners
- synthetic demo checks

Status:
- current automated test suite passes: `42 passed`

Why this mattered:
- a bad benchmark result is only meaningful if the code is mechanically correct

### 2. First real-model pilot

We started with the easiest realistic pair:
- source: `Qwen/Qwen2.5-0.5B-Instruct`
- target: `Qwen/Qwen3-0.6B`

We used:
- `600` calibration prompts in [data/calibration.txt](/Users/sujeethjinesh/Desktop/LatentWire/data/calibration.txt:1)
- `100` GSM8K math problems in [data/gsm8k_100.jsonl](/Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_100.jsonl:1)

Why this pair:
- same family
- similar size
- lowest-risk place to ask the main question: does the method work at all on real models?

### 3. Control-pair follow-up tests

After fixing real bugs in calibration and prefix alignment, we ran:
- low fusion-gate sweeps
- full precision vs quantized transfer
- dense all-layer transfer

Why:
- the "gate" is just a dial for how much translated memory to trust
- low gate values test "small hint" transfer
- full precision tells us whether quantization is hurting

### 4. Full control suite

We then ran a more systematic control suite with these axes:
- dense interpolation vs sparse `CKA` layer matching
- full precision vs quantized transfer
- low gate sweep: `0.15`, `0.25`, `0.30`
- plain prompt vs chain-of-thought source prompting
- `text+KV hybrid`

Why:
- `CKA` asks: should we send only the layers that line up best?
- sparse transfer asks: is "less but cleaner" better than "more but noisier"?
- quantization asks: can we compress the transfer heavily without losing the win?

## Main Results

## A note on scale

These scores are on a `100`-question slice.
- `0.04` means `4/100` correct
- `0.05` means `5/100` correct
- `0.06` means `6/100` correct

So the current gains are real enough to be interesting, but still small.

### Early real-model result

Before the code fixes, the method looked broken:
- target alone: `0.04`
- text-to-text: `0.10`
- RotAlign-KV: `0.00`

Takeaway:
- the first real run failed badly

### After bug fixes

The method stopped collapsing:
- best full-precision low-gate RotAlign-KV: `0.05`
- target alone: `0.04`
- text-to-text: `0.10`

Takeaway:
- there was a small positive signal
- but the method still clearly trailed the best text baseline

Reference:
- [results/followup_control_20260414/summary.md](/Users/sujeethjinesh/Desktop/LatentWire/results/followup_control_20260414/summary.md:1)

### Final control-suite result

Best result in the full control suite:
- configuration: `cka_half_seed1 / fused_quant_brief / gate=0.15`
- score: `0.06`
- transmitted bytes: about `1.2 MB`

Important comparison points:
- best dense full-precision run: `0.05` at about `19.0 MB`
- best sparse quantized run: `0.06` at about `1.2 MB`
- target alone: `0.04`

Takeaway:
- the best result so far is **sparse**, **compressed**, and **low-gate**
- this is the first control result that looks genuinely positive
- however, the improvement is still small

Important caveat:
- this `0.06` row was selected by sweeping gate values on the same
  `100`-example slice it was reported on
- that makes it a useful directional result, but not yet a clean headline number
- the next rerun should tune gates on
  [data/gsm8k_gate_search_30.jsonl](/Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_gate_search_30.jsonl:1)
  and report only on
  [data/gsm8k_eval_70.jsonl](/Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_eval_70.jsonl:1)

Reference:
- [results/control_suite_20260414_125853/latest_summary.md](/Users/sujeethjinesh/Desktop/LatentWire/results/control_suite_20260414_125853/latest_summary.md:1)

## What Helped vs What Hurt

### Helped

- **Low gate values**
  - small amounts of translated memory help more than aggressive fusion

- **Sparse `CKA` selection**
  - sending only some matched layers can be better than sending everything

- **The right seed**
  - `seed=1` mattered; seed choice is affecting outcomes

- **Compression can work**
  - surprisingly, the current best row is compressed, not full precision

### Hurt

- **Dense all-layer transfer**
  - often added noise instead of useful signal

- **Source-side chain-of-thought prompting**
  - did not help in the current setup

- **Text+KV hybrid**
  - mostly weak so far

## What Story Is Emerging

The story is no longer:
- "the method does not work"

It is now:
- "the method seems to work a little, but only in a narrow regime"

The current best explanation is:
1. passing all internal memory is too noisy
2. selecting only the best-matching layers helps
3. small amounts of transferred state help more than large amounts
4. the system is sensitive to calibration details and random seed

That is a plausible research story, but not yet a paper-winning one.

## Where We Stand For Publication

### Good news

- the implementation is real
- the code is tested
- the control pair now has a positive signal
- the systems story is starting to look interesting because the best row is both better and much cheaper

### Not good enough yet

- the gain is still small
- the result has not clearly beaten the strongest earlier text baseline (`0.10`)
- we do not yet have C2C or KVComm comparisons
- we do not yet have the full benchmark matrix
- we do not yet have a stable, repeatable result across seeds and settings

## Immediate Next Step

The next experiment should be:
- rerun the control pair with **held-out gate search**
- compare `text-to-text` under `plain`, `brief_analysis`, and `cot`
- compare fused KV, translated-only KV, and `text+KV hybrid` under the same prompting
- focus first on the best-looking branch:
  - `cka_half_seed1`
  - low gates
  - quantized and no-quantized variants
- then add one knowledge task before widening the model matrix

Why:
- the control suite suggests the method is real but fragile
- the next run has to separate a real gain from a small-slice selection effect
- the paper story also needs to show that the gain is stronger on reasoning than on knowledge

## Bottom Line

The project has moved from:
- "probably broken"

to:
- "promising but fragile"

We now have the first credible sign that direct internal-state transfer can help on a real reasoning task.
But we are still closer to **validated prototype** than to **ICLR-ready paper**.
