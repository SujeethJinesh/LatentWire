# Systems Rate And Assumption Frontier References

- date: `2026-04-30`
- purpose: primary-source positioning for the source-private systems/rate
  frontier, with explicit non-claims against KV/cache and prompt-compression
  methods.

## Sources And Use

### C2C / Cache-To-Cache Communication

- source: https://arxiv.org/abs/2510.03215
- blocker helped: closest high-rate internal-state communication comparison.
- mechanism suggested: compare access assumptions explicitly: dense cache/KV
  exchange versus endpoint-visible packet transfer.
- next experiment change: no immediate Mac-local gate; include as cross-model
  internal-state baseline when GPU/cache access is available.
- role: baseline / related-work threat.

### KVComm / KV Sharing

- source: https://openreview.net/forum?id=F7rUng23nw
- blocker helped: reviewer question about multi-agent communication through
  cache tensors.
- mechanism suggested: selective KV sharing is a native systems baseline, but
  it assumes KV access rather than source-private byte packets.
- next experiment change: keep as assumption-separated comparator until native
  cache experiments are possible.
- role: baseline / systems framing.

### TurboQuant

- source: https://arxiv.org/abs/2504.19874
- blocker helped: latest low-bit vector/KV quantization comparison.
- mechanism suggested: online quantized transport is a useful byte/latency
  neighbor, not a task-evidence packet baseline under endpoint-only access.
- next experiment change: add byte-floor and non-claim rows; defer native
  TurboQuant comparison to GPU/server setting.
- role: systems baseline / ablation inspiration.

### QJL

- source: https://arxiv.org/abs/2406.03482
- blocker helped: reviewer concern that one-bit sketches may dominate the
  systems story.
- mechanism suggested: sign-sketch / Johnson-Lindenstrauss-style compression
  should be reported as a KV byte-floor neighbor, and may inspire future
  learned packet projections.
- next experiment change: keep QJL-style byte floor in the systems frontier;
  do not call it an accuracy baseline without native KV runs.
- role: systems baseline / mathematical inspiration.

### KIVI And KVQuant

- sources: https://arxiv.org/abs/2402.02750 and https://arxiv.org/abs/2401.18079
- blocker helped: strong same-model KV-cache compression alternatives.
- mechanism suggested: asymmetric and low-bit KV layouts clarify why our
  contribution is not generic cache compression.
- next experiment change: report only assumption-aware byte floors on current
  hardware.
- role: systems baseline.

### LLMLingua And LLMLingua-2

- sources: https://aclanthology.org/2023.emnlp-main.825/ and
  https://arxiv.org/abs/2403.12968
- blocker helped: prompt-compression baseline threat.
- mechanism suggested: include same-byte and query-aware text relays to show
  when visible text compression does and does not explain the packet gain.
- next experiment change: keep same-byte structured text and query-aware
  diagnostic-span rows in every headline rate table.
- role: prompt-compression baseline.

### Gist Tokens

- source: https://arxiv.org/abs/2304.08467
- blocker helped: learned context compression comparison.
- mechanism suggested: learned soft/prefix compression is a related
  high-capacity setting, but it requires training/interface assumptions beyond
  endpoint byte packets.
- next experiment change: include as related-work row; consider only if we add
  learned soft-token receiver experiments.
- role: baseline / framing.

## Bottom Line

The frontier should be written as a rate-and-assumption result: LatentWire shows
source-private task evidence at extreme byte rates under destructive controls.
It should not claim native superiority over KV compression, C2C, KVComm, or
prompt compression until those methods are run in their own fair access model.
