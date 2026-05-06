# HybridKernel Phase 0 Setup Check

- date: 2026-05-05
- status: historical_initial_setup_snapshot_superseded
- scope: local-only setup and tiny config fetches; no SSH, no GPU, no global installs,
  no model weights, and no large repository clones.

This file records the initial Phase 0 setup snapshot. It is no longer the
current runtime surface for HybridKernel correctness work. Current Mac
reproducibility is recorded in `phase0/local_preflight.md` and
`phase2/mac_reproducibility_command.md`: the repo-local `./venv_arm64` imports
PyTorch 2.6.0 and Triton from the repo-local `triton-cpu` source checkout, and
the Phase 3/4 reference plus Triton-interpreter checks pass there.

## Environment

Initial project-local environment:

```text
experimental/hybridkernel/.venv
Python 3.9.13
```

Initial import check result for `experimental/hybridkernel/.venv`:

```text
torch: missing in initial per-project venv
transformers: missing in initial per-project venv
numpy: missing in initial per-project venv
pytest: missing in initial per-project venv
huggingface_hub: missing in initial per-project venv
triton: missing in initial per-project venv
```

This initial per-project environment was enough for setup provenance and config
fetching only. It should not be used to judge current HybridKernel readiness.
Use `./venv_arm64` and the command in `phase2/mac_reproducibility_command.md`
for the current Mac-local reference, checker, and Triton-interpreter gates.

## Configs Fetched

Only `config.json` files were downloaded from Hugging Face. No model weights
were downloaded.

| Model | Local file | Status | SHA256 |
|---|---|---|---|
| `ibm-granite/granite-4.0-h-tiny` | `phase0/configs/ibm-granite-4.0-h-tiny.config.json` | fetched | `bda8fd574ace7d968d82397f59ea6b9a702a077bbeab279a65b9dad7386a82c6` |
| `ibm-granite/granite-4.0-h-small` | `phase0/configs/ibm-granite-4.0-h-small.config.json` | fetched | `8616e9f0b30e6fac9696f7c1e1dbd08f1a850ac4af0de6353f7d6009043702ae` |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | `phase0/configs/qwen3-next-80b-a3b-instruct.config.json` | fetched | `2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303` |
| `nvidia/Nemotron-H-8B-Base` | none | blocked by HTTP 401 or unavailable public path | n/a |
| `nvidia/Nemotron-3-Nano-30B-A3B` | none | blocked by HTTP 401 or unavailable public path | n/a |
| `ServiceNow-AI/Apriel-H1-15B-Thinker` | none | blocked by HTTP 401 or unavailable public path | n/a |

## Config Observations

Granite 4.0 H Tiny and Small both expose `model_type=granitemoehybrid`,
`num_hidden_layers=40`, and an explicit `layer_types` list. The pattern has
four attention layers at indices 5, 15, 25, and 35, with Mamba layers elsewhere.
That gives eight immediate layer-type boundaries per forward pass:

```text
mamba->attention: 4
attention->mamba: 4
```

Qwen3-Next exposes `model_type=qwen3_next`, `num_hidden_layers=48`, and
`full_attention_interval=4`, plus linear-attention/GDN-style dimensions. This is
useful as a config-only hybrid reference, but it is not the same Granite/Nemotron
Mamba2 boundary surface.

## Phase 0 Verdict

The initial Phase 0 snapshot was partial because dependencies were not installed
in `experimental/hybridkernel/.venv` and several target configs required gated
or corrected access. The current repo-level Mac setup supersedes that snapshot
for correctness work; native GPU performance evidence is still unavailable.
