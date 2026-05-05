# SVAMP32 C2C MPS Generation Trace Syndrome Probe Manifest

- date: `2026-05-05`
- status: `c2c_mechanism_syndrome_probe_fails_gate`
- pass gate: `false`

## Summary

After repairing local MPS C2C compatibility, the replayed dense teacher reaches
`16/32` on SVAMP32 and exposes `10` teacher-only rows. However, the
generation-summary trace syndrome decoder still fails as a deployable
distillation path:

- feature matrix: `32 x 10080`, `float32`;
- matched: `13/32`;
- zero-source: `14/32`;
- target-only: `14/32`;
- label-shuffled: `15/32`;
- clean teacher-only recovered by matched: `4`;
- clean source-necessary IDs beyond controls: `0`.

The result keeps the C2C-distillation story alive at the teacher level but
rules out the current ridge decoder over generation-summary traces as the
positive sparse-packet method.

## Artifact Hashes

| Artifact | SHA256 |
|---|---|
| `generation_trace_probe.json` | `5e6990585ef776c9363750c1e6a7d61ad78d6cb335ba43942f6944ba6499b169` |
| `generation_trace_probe.md` | `1df3be0b596fd2e250f551ea55714997f3ea6b554bae4bb4d0979cbac57fec6c` |
