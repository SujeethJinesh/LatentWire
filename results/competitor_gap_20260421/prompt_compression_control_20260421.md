# LLMLingua-Style Prompt Compression Control

This control is deterministic and does not load the LLMLingua model stack.
It uses a lexical token-budget proxy over the existing GSM8K / SVAMP prompt data
to separate prompt-budget effects from learned compressor effects.

Budget ratio: `0.50`
Minimum budget: `16` token-proxy units

| Source | N | Original chars | Original tokens proxy | Compressed budget | Compressed tokens proxy | Number preservation | Answer-span coverage | Answer-span preservation | Est. bytes saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k_eval_70.jsonl | 70 | 348.4 | 74.5 | 37.5 | 37.5 | 1.00 | 5.7% | 1.00 | 123.5 |
| svamp_eval_70.jsonl | 70 | 175.9 | 37.9 | 19.7 | 19.7 | 1.00 | 2.9% | 1.00 | 71.5 |

## Method

- Tokens are approximated with a regex word / number / punctuation proxy, not a model tokenizer.
- Compression keeps high-salience lexical items, all numeric tokens, and any answer span that is actually present in the prompt.
- The goal is a fair no-download control for LLMLingua-style prompt compression, not a replacement for learned prompt compression.

## Claim Risks

- This is a lexical baseline, so it cannot support claims about LLMLingua's learned ranking quality.
- Answer-span preservation is only meaningful on examples where the answer string appears in the source prompt.
- Token-budget values are proxy counts, so they are descriptive rather than tokenizer-exact.
- Accuracy is not measured here; this control only measures compression and preservation telemetry.
