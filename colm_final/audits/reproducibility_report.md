# Reproducibility Report

Date: 2026-05-02

## Status

- Frozen artifact hash validation: passed, 32/32 manifest entries matched.
- Repository test suite: passed, `1324 passed in 145.02s`.
- Targeted COLM test subset from repo root: passed, `16 passed in 18.84s`.
- Targeted COLM test subset from `colm_final/code`: passed, `16 passed in 1.84s`.
- LaTeX build from `colm_final/paper`: passed, 9-page PDF, no unresolved
  references/citations and no overfull boxes.

## Exact Commands

Run from the repository root unless noted.

Full tests:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python -m pytest -p no:cacheprovider tests -q
```

Targeted COLM tests:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python -m pytest -p no:cacheprovider \
  tests/test_build_source_private_arc_challenge_fourier_anchor_syndrome_gate.py \
  tests/test_build_source_private_arc_challenge_seed_stability.py \
  tests/test_build_source_private_arc_challenge_source_family_cache_falsification.py \
  tests/test_analyze_source_private_arc_cross_family_failure_decomposition.py \
  tests/test_build_source_private_systems_boundary_figure_table.py \
  tests/test_build_source_private_arc_challenge_candidate_syndrome_connector_gate.py \
  tests/test_build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate.py -q
```

Bundle-local targeted COLM tests:

```bash
cd colm_final/code
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ../../venv_arm64/bin/python -m pytest -p no:cacheprovider \
  tests/test_build_source_private_arc_challenge_fourier_anchor_syndrome_gate.py \
  tests/test_build_source_private_arc_challenge_seed_stability.py \
  tests/test_build_source_private_arc_challenge_source_family_cache_falsification.py \
  tests/test_analyze_source_private_arc_cross_family_failure_decomposition.py \
  tests/test_build_source_private_systems_boundary_figure_table.py \
  tests/test_build_source_private_arc_challenge_candidate_syndrome_connector_gate.py \
  tests/test_build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate.py -q
```

Paper build:

```bash
cd colm_final/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
```

Frozen artifact hash validation:

```bash
./venv_arm64/bin/python - <<'PY'
import json, pathlib, hashlib
root = pathlib.Path("colm_final/evidence/results")
fail = []
ok = 0
for manifest in sorted(root.glob("*/manifest.json")):
    data = json.loads(manifest.read_text())
    base = manifest.parent
    if "artifact_sha256" in data:
        entries = list(data["artifact_sha256"].items())
    elif "sha256" in data:
        entries = list(data["sha256"].items())
    elif "files" in data:
        entries = [
            (pathlib.Path(row["path"]).name, row["sha256"])
            for row in data["files"]
            if isinstance(row, dict)
        ]
    else:
        entries = []
    for rel, expected in entries:
        path = base / pathlib.Path(rel).name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            ok += 1
        else:
            fail.append((str(path), actual, expected))
print(f"manifest_hash_entries_ok={ok}")
print(f"manifest_hash_entries_failed={len(fail)}")
for row in fail:
    print("FAIL", row)
PY
```

## Paper-Row Rerun Commands

These reproduce metrics from cached, answer-key-forbidden source-choice
artifacts. They are not byte-for-byte deterministic because JSON records include
creation timestamps and some artifacts include measured local latency fields.

ARC headline:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_arc_challenge_fourier_anchor_syndrome_gate.py \
  --output-dir .debug/repro_arc \
  --seeds 47,53,59,61,67,71,73,79,83,89 \
  --budget-bytes 8 \
  --bootstrap-samples 2000
```

OpenBookQA headline:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_arc_challenge_seed_stability.py \
  --output-dir .debug/repro_obqa \
  --train-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl \
  --eval-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_test.jsonl \
  --anchor-predictions results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/predictions.jsonl \
  --split-name openbookqa_test_hashed_3b \
  --seeds 47,53,59,61,67 \
  --budget-bytes 3 \
  --feature-dim 384 \
  --code-dim 96 \
  --feature-mode hashed \
  --bootstrap-samples 500
```

Phi-3 falsification:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_arc_challenge_source_family_cache_falsification.py \
  --output-dir .debug/repro_phi3 \
  --skip-cache-materialization \
  --alt-validation-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_validation/source_prediction_cache.jsonl \
  --alt-test-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_test/source_prediction_cache.jsonl \
  --alternate-source-family phi3_mini_4k \
  --budget-bytes 8 \
  --seeds 47,53,59,61,67,71,73,79,83,89 \
  --bootstrap-samples 2000
```

Systems boundary:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_systems_boundary_figure_table.py \
  --output-dir .debug/repro_systems
```

Failure decomposition:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python \
  scripts/analyze_source_private_arc_cross_family_failure_decomposition.py \
  --output-dir .debug/repro_decomp
```

## Rerun Results This Pass

- OpenBookQA rerun completed: packet mean `0.3784`, target `0.276`, same-byte
  text `0.350`, minimum paired CI low `+0.038`, 5/5 seeds pass.
- Systems rerun completed: pass gate true, max headline packet `11B`,
  one-token 1-bit-per-KV-element floor `768B`, ratio `69.818x`.
- Failure decomposition rerun completed: selected next gate
  `common_feature_connector_with_stronger_source`.
- Exact ARC and Phi-3 10-seed/2000-bootstrap reruns were started but stopped
  after more than 30 minutes without output files; frozen artifact hashes are
  therefore the authoritative verification for those rows in this pass.

## Key Frozen Hashes

| Artifact | SHA256 |
|---|---|
| `paper/latentwire_colm2026.pdf` | `6bec5e8e355bc973abd37d5555d4c1bf0a3be5f707e2b78242eb72dc31b37ad6` |
| ARC headline JSON | `45b103c9330e4b512c18d6572d2915787a7caeb9c3aa8528e045a06292146f17` |
| ARC matched predictions | `9ab7b9da8a377c59bd206906b02fb26baecce1df433925aa1c34f172ac124c33` |
| OpenBookQA headline JSON | `999fefa6cebb762eebbb78957969fc2832781ec79734aefdaa1746227dceaec6` |
| Phi-3 falsification JSON | `9a7cfc159d77dc583d4294773c6ed595ce3f1c3ed544bad4cfb556fff88626fe` |
| Failure decomposition JSON | `ce9814b893940739d755c0aa17761e5f1688621e9a5414cae3e800cfb34d3f13` |
| Systems boundary JSON | `69774ee3d138971b55fc83b506a80ba5debdf606857570a4b83eebd6554a1ed1` |

## Remaining Reproducibility Risk

The package is metric-reproducible and hash-verifiable for frozen artifacts,
but a reviewer may still ask for a deterministic compare mode that strips
timestamps and latency fields from rerun JSON before diffing.
