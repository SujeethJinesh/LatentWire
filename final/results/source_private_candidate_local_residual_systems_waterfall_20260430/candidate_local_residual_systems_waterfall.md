# Candidate-Local Residual Systems Waterfall

- pass gate: `True`
- n512 packet rows passing: `9/9`
- packet record bytes at 8B payload: `11`
- batch-64 line bytes/request: `11.00`
- max current Python packet p50: `0.303916` ms/request
- max resident sparse decode p50: `5.23193` us/request
- max cold candidate feature build: `4.16122` ms/request
- source text exposed: `False`
- source KV exposed: `False`

## Checks

| Check | Pass | Value |
|---|---:|---:|
| `all_n512_packet_rows_pass` | `True` | `9/9` |
| `source_private_exposure` | `True` | `source_text_exposed=false, source_kv_exposed=false` |
| `calibration_eval_exact_id_overlap_zero` | `True` | `0` |
| `transformed_surface_overlap_zero` | `True` | `0` |
| `resident_sparse_decode_exact_if_measured` | `True` | `0` |

## Interpretation

This artifact separates the live candidate-local residual receiver into packet boundary accounting, current Python nonresident decode, optional Mac resident sparse decode over cached public candidate residuals, and source text/KV exposure. It is a Mac-local systems trace, not a production vLLM or NVIDIA serving claim.

## Non-Claims

- No HBM, PCIe, NVLink, TPOT, goodput, or production serving counter is measured here.
- The receiver cache is public candidate state and is reported separately from source-private packet bytes.
- C2C/KVComm rows remain matched baselines to run, not defeated baselines in this artifact.
