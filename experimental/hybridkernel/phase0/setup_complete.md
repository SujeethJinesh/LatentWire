# HybridKernel Phase 0 Setup Check

- date: 2026-05-05
- status: partial_mac_setup_complete_for_audit
- scope: local-only setup and tiny config fetches; no SSH, no GPU, no global installs,
  no model weights, and no large repository clones.

## Environment

Project-local environment:

```text
experimental/hybridkernel/.venv
Python 3.9.13
```

Import check result:

```text
torch: missing
transformers: missing
numpy: missing
pytest: missing
huggingface_hub: missing
triton: missing
```

I did not install the full requirements stack because the current task only
needs setup provenance and a quick source audit. Installing Torch/Triton on Mac
would be slower and is not needed until Phase 3/4 reference work.

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

Phase 0 is not fully complete because dependencies are not installed and several
target configs require gated or corrected access. It is complete enough for the
requested quick Phase 1 audit and for a Phase 2 Granite-only architecture map.
