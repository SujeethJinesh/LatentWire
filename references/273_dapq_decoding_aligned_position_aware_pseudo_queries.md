# DapQ

- Title: `Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries`
- Date: 2026-03-12
- Link: https://arxiv.org/abs/2603.11564
- Why it matters here:
  - strongest recent decoding-aligned query-aware control for the same general question of which KV states matter at inference time
  - useful if a public repo appears and we want a sharper mechanistic control than generic compression or pruning

Most transplantable mechanism:
- drive cache selection with pseudo-queries aligned to the actual decoding trajectory rather than only static importance or compression heuristics

Immediate use in our setting:
- keep as the best future comparator/control if code becomes easy to replay on our GSM JSONL slices
