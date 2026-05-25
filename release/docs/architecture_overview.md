# Architecture Overview

The release package is intentionally small. The source tree contains only the
interfaces needed to reproduce paper claims:

- `data.py` loads YAML configs and stores frozen expected claim values.
- `models.py` validates model metadata from configs.
- `quantization.py` implements minimal symmetric INT4 quantization and
  protected-channel masks.
- `methods/` contains pure mask builders for static, M11b, M26, DecDEC, and a
  ParoQuant rotation helper.
- `metrics.py` implements recovery, set-leaving, KL, and bootstrap helpers.
- `analysis.py` implements spectral entropy and autocorrelation helpers.
- `src/scripts/` contains one reproduction entry point per paper claim family.

The paper's W4A16 setup means weights are quantized to signed INT4 while
activations remain high precision. Protected channels are represented as
boolean masks that a full runner can use to bypass quantization for selected
rows or columns.

FAST_VERIFY mode checks frozen expected outputs. Full reproduction keeps the
same script surface but replaces the frozen replay with model loading,
activation capture, quantization, and scoring.
