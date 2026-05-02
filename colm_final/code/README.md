# COLM Code Snapshot

This directory contains a Python snapshot of the repository scripts and tests
needed to audit the COLM bundle.

Recommended commands are run from the repository root, because the experiment
scripts intentionally resolve default data and result paths relative to the
repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python -m pytest -p no:cacheprovider tests -q
```

The copied code can also run the targeted COLM test subset from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ../../venv_arm64/bin/python -m pytest -p no:cacheprovider \
  tests/test_build_source_private_arc_challenge_fourier_anchor_syndrome_gate.py \
  tests/test_build_source_private_arc_challenge_seed_stability.py \
  tests/test_build_source_private_arc_challenge_source_family_cache_falsification.py \
  tests/test_analyze_source_private_arc_cross_family_failure_decomposition.py \
  tests/test_build_source_private_systems_boundary_figure_table.py \
  tests/test_build_source_private_arc_challenge_candidate_syndrome_connector_gate.py \
  tests/test_build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate.py
```

For exact paper-row reproduction commands and expected hashes, see
`../audits/reproducibility_report.md`.
