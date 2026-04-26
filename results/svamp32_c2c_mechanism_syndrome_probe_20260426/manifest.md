# SVAMP32 C2C Mechanism Syndrome Probe Manifest

- date: `2026-04-26`
- git commit at run time: local working tree, pre-commit
- scale-up rung: strict small gate
- status: `fails_gate`
- device: `mps`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- target set: `results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.json`

## Runs

### Prefill Scalar Trace

- artifact: `prefill_scalar_trace_targetpool_probe.json`
- sha256: `0220017a3a76dff54056fa8caeefde244b61a87cde27bfcbcab93c67f081c902`
- markdown: `prefill_scalar_trace_targetpool_probe.md`
- markdown sha256: `cf51af65b4700f80656b5b16739c42e15a214ec349f7688c1fbb395998044c20`
- feature dim: `336`
- result: matched `11/32`, target-only `14/32`, zero-source `14/32`,
  shuffled-source `9/32`, label-shuffle `14/32`, slots-only `8/32`,
  clean source-necessary `0/6`

### Prefill Residual Trace

- artifact: `prefill_residual_trace_targetpool_probe.json`
- sha256: `685d76e3640b17084b25544c970ec8a95efe1555e5d36469fb49ba88325176f7`
- markdown: `prefill_residual_trace_targetpool_probe.md`
- markdown sha256: `d5ab8c2dbbf68e18258001de7dec69b735288f7627b87f17e7030aa0e3193595`
- feature dim: `896`
- feature tensor sha256: `75ad00f84a99ae632ec5641fa53e66e987188ba693858079dfae319381de7e73`
- result: matched `12/32`, target-only `14/32`, zero-source `13/32`,
  shuffled-source `9/32`, label-shuffle `14/32`, slots-only `8/32`,
  clean source-necessary `0/6`

## Decision

The C2C mechanism-summary branch fails the strict SVAMP32 small gate. Residual
summaries improve scalar traces by only one matched row and remain below the
target-only decoder floor, with no clean source-necessary recovery.

Do not scale this summary-feature variant. A future C2C-derived branch needs a
crisper new reason, such as token/layer-level residual coding with held-out
features and a stronger anti-cache control, before spending compute.
