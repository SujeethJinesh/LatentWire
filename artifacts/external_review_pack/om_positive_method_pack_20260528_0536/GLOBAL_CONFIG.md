# Global Config

- Repo commit: `dde41fbf3ebffc441ecf15a0badadd97b3eadf0a`
- Branch: `main`
- Python/env: see `provenance/environment.txt`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`
- Models: see `provenance/model_checkpoints.csv`
- Prompt sets: AIME-2025 deterministic indices 0-11/0-23 depending on packet; smoke traces are fixed in `artifacts/funnel_prefilters/smoke_traces.json`.
- Quantization: W4A16 symmetric per-output-channel int4 unless a packet states ParoQuant rotation.
- Recovery definition: `1 - (PPL_method - PPL_BF16) / (PPL_static_1pct - PPL_BF16)`.
- No-gap handling: traces with `PPL_static_1pct <= PPL_BF16` are marked no-gap and excluded from median recovery.
- Bootstrap: packet-specific percentile/BCa fields are copied as reported; see `tables/bootstrap_ci_summary.csv`.
