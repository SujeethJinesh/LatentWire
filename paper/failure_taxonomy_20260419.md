# Failure Taxonomy / Blocker Evidence (2026-04-19)

| Blocker | Observed Symptom | Best Supporting Result | Best Null / Contrast | What It Rules Out | Next Implication | Status |
|---|---|---|---|---|---|---|
| Head-space symmetry / permutation mismatch | Raw grouped transport and hard grouped permutation both stay weak on exact Qwen GSM70 | `grouped_permutation = 0.0286` | `fixed prior = 0.0857` | Simple head reassignment is not enough | Use richer transport or canonicalization, not just hard matching | Open |
| Geometry-aware transport is only partially helpful | Signature and subspace transport improve over naive grouped transport but plateau at `0.0429` | `grouped_signature_transport = 0.0429`, `grouped_subspace_transport = 0.0429` | `grouped_transport = 0.0143` | Better grouped cost helps, but only a little | If transport keeps improving, it must be richer than current grouped penalties | Partial |
| Canonical basis alone is not the rescue | Low-rank grouped canonical transport falls to `0.0286` | `grouped_canonical_transport = 0.0286` | `grouped_signature_transport = 0.0429` | “Just canonicalize first” is too weak | Transport-plus-correction is more promising than canonicalization-only | Open |
| Covariance geometry is not the next shortcut | Covariance-aware transport + rank-4 residual collapses to `0.0143` | `grouped_covariance_resid4 = 0.0143` | `grouped_subspace_resid4 = 0.0571` | Covariance shape alone is not the right geometry in this regime | Stop spending time on covariance-aware grouped transport | Open |
| Correction alone is not enough | Learned affine and learned head-ridge both collapse to `0.0000` | `learned_affine = 0.0000`, `learned_head_ridge = 0.0000` | `fixed prior = 0.0857` | Small learned correction does not rescue a weak transport map | If correction matters, it has to sit on top of a better transport | Open |
| Transport plus tiny correction can help locally | Rank-4 residual on grouped subspace transport lifts GSM70 from `0.0429` to `0.0571` | `grouped_subspace_resid4 = 0.0571` | `grouped_subspace_transport = 0.0429` | Transport-plus-correction is not dead | This is the best remaining internal lane | Partial |
| K/V asymmetry is real | `K-only` works better than `V-only`; translated-only collapses | `K-only = 0.0571` on best historical branch, `V-only = 0.0000` | `translated-only = 0.0000` | Full KV transfer is not equally useful | Keep K-centric story and controls in the paper | Partial |
| Query/blind selection confound must be controlled | Fixed prior beats shuffled prior on the main same-pair GSM70 split | `fixed prior = 0.0857`, `shuffled fixed prior = 0.0429` | Shuffled-prior null | Gains are not just any uneven sparsity mask | Retain blind-selector nulls in main tables | Partial |
| Calibration fit does not imply held-out task gains | Several transport branches fit calibration better than fixed prior but lose on GSM70 | e.g. `grouped_transport` calibration much better, GSM70 `0.0143` | `fixed prior = 0.0857` | Offline fit is not the right success metric | Judge every branch on held-out task behavior, not calibration quality | Open |
| External baseline gap remains the main paper risk | `C2C` stays ahead on GSM70, GSM100, and SVAMP70 | `C2C = 0.1286` on GSM70, `0.1100` on GSM100, `0.4429` on SVAMP70 | Best internal branches `0.0857`, `0.0700`, `0.1714` | We do not yet have a positive main-method result | Either beat `C2C` or narrow the paper to blocker/mechanism | Open |

## Current Read

- Strongest internal positive clue: **grouped subspace transport + rank-4 residual**.
- Strongest external bar: **`C2C`**.
- Strongest honest paper framing today: **blocker/mechanism paper with one live transport-plus-correction lane**, not a broad positive method claim.
