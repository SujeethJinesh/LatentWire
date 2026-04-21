# LLMLingua-Style Prompt Compression Control

Date: 2026-04-21

This note documents the no-download control used for LatentWire's prompt-compression comparator. The point is to separate a *learned* LLMLingua/LongLLMLingua compressor from a deterministic lexical budget baseline that can be run locally on GSM8K / SVAMP prompt data.

Primary sources:

- [LLMLingua paper](https://aclanthology.org/2023.emnlp-main.825/)
- [LongLLMLingua paper](https://aclanthology.org/2024.acl-long.91/)
- [LLMLingua GitHub](https://github.com/microsoft/LLMLingua)

Mechanism summary:

- LLMLingua uses a compact language model to score and prune prompt tokens under a target token budget.
- LongLLMLingua extends the idea to long-context settings and explicitly addresses middle-loss / position-bias issues.
- The local repo clone at `references/repos/LLMLingua` confirms the intended API shape, but this control does not require loading those model weights.

Control used here:

- Deterministic lexical token-budget analysis over `data/gsm8k_eval_70.jsonl` and `data/svamp_eval_70.jsonl`.
- Regex token proxy, not a model tokenizer.
- Keeps high-salience lexical items, numeric spans, and answer spans when the answer string is actually present in the prompt.
- Reports original chars/tokens proxy, compressed budget, number preservation, answer-span preservation when available, estimated bytes saved, and claim risks.

Why this is the right fallback:

- It gives a paper-safe compression control without large model downloads.
- It isolates prompt-length effects from learned compressor effects.
- It is deterministic, so the artifact can be regenerated and diffed exactly.

Claim-risk note:

- This baseline does not support claims about learned token ranking or end-to-end QA accuracy.
- It is a budget/preservation diagnostic only, not a replacement for LLMLingua or LongLLMLingua evaluation.
