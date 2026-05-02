# ARC Llama-8B Failure Probe

- pass gate: `False`
- selected router: `source_matches_llama_prediction`
- selected router deployable without source index: `False`
- best deployable router: `packet_margin_ge:0.131799`
- validation selected-router accuracy: `0.490278`
- test selected-router accuracy: `0.441860`
- validation best-deployable-router accuracy: `0.408333`
- test best-deployable-router accuracy: `0.361522`
- test Llama/Qwen oracle accuracy: `0.532347`
- test source/Qwen oracle accuracy: `0.613108`
- test same-byte-text minus Llama: `0.126427`
- test source-to-packet loss: `0.185624`

## Lay Explanation

This probe checks whether the failed Llama row failed because the source model was bad, because the 12-byte packet lost useful source choices, or because a simple rule can tell when to trust Llama instead of Qwen.

## Decision

The Llama scout contains diagnostic conditional headroom, but no reviewer-safe deployable source-choice method. The best router overall uses the audit-only source-selected index, the best packet-observable router is a small margin rule, and same-byte visible text remains stronger than the Llama packet on test. Treat this as evidence that the source answer signal exists but the current packet codec is lossy; move to a learned query/cache or soft-prefix connector rather than another source-choice sender.
