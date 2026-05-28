# Omitted Large Artifacts

- Full model weights are omitted. They are recoverable from Hugging Face model
  IDs and snapshot commits in `provenance/model_checkpoints.csv`.
- Full activation tensors are omitted. The V1 decision only needs score caches,
  per-trace metrics, and checker output, all included here.
- Full BF16 token traces are not needed for numeric verification; the trace
  manifest is included in `experiments/v1_paroquant_nemotron/traces_used.json`.
