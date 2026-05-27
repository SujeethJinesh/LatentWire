# Math Formalization Pass

Timestamp: 2026-05-27T10:45Z

## Decision

The math and mechanism formalization pass was completed without triggering an
override condition.

## Added Formalization

- Defined the channel-score model
  $\sigma_{\ell,c}(t)=\mathbb{E}_{p}[|a_{\ell,p,c}(t)|]$ and the ideal
  top-$K$ protected set $S_{\ell}(t)$ before the strict set-leaving equation.
- Added explicit KL functional forms for sublinear square-root, linear, and
  superlinear fits.
- Added an AR(1) residual equation for the KL fit residuals.
- Added a simple error-allocation model linking set-leaving to wasted protected
  capacity.
- Recast the three mechanisms as boundary discontinuity, insufficient smoothed
  top-1% budget, and budget binding.
- Added an appendix R^2 table for the Granite-Small KL fits.

## Verification

The existing KL packet supports the added formalization:

| Regime | R2 sqrt | R2 linear | R2 t^2 | AR(1) rho |
|---|---:|---:|---:|---:|
| Static top-1% | 0.469 | 0.454 | 0.380 | 0.515 |
| DecDEC proxy | 0.409 | 0.379 | 0.298 | 0.446 |
| M11 EMA | 0.399 | 0.368 | 0.284 | 0.439 |

The square-root fit remains best for all three regimes; no compound-error
override was triggered.

The paper rebuilt successfully with TeX warnings only. `pypdf` reports 14 total
pages, with references beginning on page 8, so the main body remains within the
8-page target.

## Next Queue Item

Proceed to the baseline-vetting queue before new positive-method mechanisms:
V1 ParoQuant on Nemotron, then V2 M11b top-10 on DeepSeek and Falcon, subject to
the remaining GPU budget and stop thresholds.
