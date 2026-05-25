# Verification Log

Verification date: `2026-05-25`

Verified commit: `a150edee3abd3f85f13c5defb2d5e7a79b466eb5`

Mode: FAST_VERIFY. The release scripts now fail explicitly outside
`--fast-verify` or `--dry-run`; full-fidelity GPU reproduction remains in the
archived experimental packet runners and is not implemented in `release/`.

## 1. Starting State

Clean directory: `/tmp/repro_verification_1`

Command:

```bash
git clone https://github.com/SujeethJinesh/LatentWire.git repo
```

Starting environment:

- OS: Linux `6.8.0-110-generic`
- Python: `3.12.3`
- GPU visible: NVIDIA RTX PRO 6000 Blackwell Server Edition, `97887 MiB`
- Disk at clone path: `33G` available on `/`
- Clone source: GitHub origin, not local copy

The clone contained no pre-existing virtual environment and no downloaded
model checkpoints.

## 2. Venv Setup

README install commands were executed exactly:

```bash
cd release
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.3
python -m pip install -e ".[dev]"
```

Result: setup succeeded. Pip installed `numpy==2.1.2`, `PyYAML==6.0.3`, and
`pytest==9.0.3` plus transitive pytest dependencies. No README deviation was
needed.

## 3. Dependency Verification

Command:

```bash
python src/scripts/verify_environment.py
```

Result: passed.

Observed package versions:

- `numpy`: `2.1.2`
- `PyYAML`: `6.0.3`

GPU check was not required in FAST_VERIFY mode. The script reported
`"gpu": false` because `--require-gpu` was not used.

## 4. Model Checkpoint Download

FAST_VERIFY intentionally does not download model checkpoints. The original
experimental packets used the configured HuggingFace models and enough disk for
checkpoints and score outputs. README now states that `release/` does not
include the full model-inference runners.

No authentication-gated checkpoint download was exercised in this fast gate.

## 5. Reproduction Scripts

Command:

```bash
bash src/scripts/reproduce_all.sh --fast-verify
```

Result: passed. The command ran every reproduction script listed in
`docs/reproducing_results.md` and wrote JSON outputs under `results/`.

Generated outputs:

- `results/set_leaving/set_leaving.json`
- `results/static_union/static_union.json`
- `results/m2/m2.json`
- `results/m10/m10.json`
- `results/m11/m11.json`
- `results/m11b/m11b.json`
- `results/m18/m18.json`
- `results/m26/m26.json`
- `results/decdec/decdec.json`
- `results/paroquant/paroquant.json`
- `results/kl_accumulation/kl_accumulation.json`
- `results/fft/fft.json`
- `results/per_component/per_component.json`
- `results/trace_heterogeneity/trace_heterogeneity.json`

Each output reports the frozen claim values and per-claim tolerance checks.

Test suite command:

```bash
python -m pytest
```

Result: `7 passed in 0.09s` on the first clean clone.

## 6. Standalone Script Verification

Three scripts were run independently after `reproduce_all.sh`:

```bash
python src/scripts/reproduce_m2.py --config configs/granite_small.yaml --fast-verify
python src/scripts/reproduce_paroquant.py --config configs/granite_small.yaml --fast-verify
python src/scripts/reproduce_kl_accumulation.py --config configs/granite_small.yaml --fast-verify
```

Result: all passed and produced their expected JSON outputs without relying on
state created by `reproduce_all.sh` beyond the installed package.

## 7. README Clarity Review

README was sufficient for FAST_VERIFY:

- project purpose identifiable in the opening paragraph,
- setup commands worked literally,
- quick-start command worked after install,
- hardware requirements state CPU fast mode and clarify that full GPU mode is
  not implemented in `release/`,
- citation and license are present.

Known limitation: README documents fast verification only. Full-fidelity
model-inference reproduction remains outside this minimal release package.

## 8. Double-Check Pass

Second clean directory: `/tmp/repro_verification_2`

Commands:

```bash
git clone https://github.com/SujeethJinesh/LatentWire.git repo
cd repo/release
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.3
python -m pip install -e ".[dev]"
python src/scripts/verify_environment.py
bash src/scripts/reproduce_all.sh --fast-verify
python -m pytest
```

Result: passed without intervention. `pytest` reported `7 passed in 0.12s`.

## 9. Summary

FAST_VERIFY reproduction succeeds from two clean GitHub clones using the README
installation flow. Every paper-claim reproduction script runs in fast mode,
and standalone script execution works for sampled scripts.

Full-fidelity GPU reproduction is not implemented or verified in this release
package. The release package must not claim complete full GPU reproduction
until dedicated full-mode runners are added and the same clean-clone process is
repeated without `--fast-verify`.
