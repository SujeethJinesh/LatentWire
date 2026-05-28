# Restricted KLLOOK oracle

- **Hypothesis:** Oracle KL lookahead exposes channel-set ceiling if cheap methods fail.
- **Method implemented:** Sampled forward-KL lookahead runner exists; no decisive restricted packet in current queue.
- **Baseline/control:** BF16 and static top-1% W4A16
- **Models/traces:** Granite/Nemotron optional
- **Result/status:** DEFERRED
- **Why it worked/failed:** Reserved for after LAMBDA/HYST smoke if ceiling evidence is needed.
- **Supporting artifacts:** No direct source artifact present
- **Caveats:** None beyond the scope notes in the source packet.
- **External reviewer should inspect:** Read summary.md, metrics.json, per_trace.csv, decision.json.
