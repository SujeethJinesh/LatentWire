"""
RotAlign-KV: cross-model KV-cache transfer via rotational alignment and
Lloyd-Max quantization.

Core pipeline:

    source KV  ->  random / Hadamard rotation   (Gaussianize)
                   optional ZCA whitening
                   Procrustes / ridge / CCA / reduced-rank alignment
                   Lloyd-Max quantization
                   inverse rotation
               ->  gated fusion into target KV
"""

from .rotation import (
    random_orthogonal,
    hadamard_matrix,
    dct_matrix,
    make_rotation,
    apply_rotation,
    kurtosis,
    verify_gaussianization,
    fit_zca_whitening,
    apply_whitening,
)
from .procrustes import (
    identity_projection,
    orthogonal_procrustes,
    orthogonal_procrustes_randomized,
    ridge_projection,
    cca_projection,
    reduced_rank_regression,
    fit_alignment,
    alignment_quality,
)
from .quantize import (
    GaussianQuantizer,
    lloyd_max_gaussian,
)
from .translator import (
    RotAlignKVTranslator,
    TranslatorConfig,
)

__all__ = [
    # rotation
    "random_orthogonal",
    "hadamard_matrix",
    "dct_matrix",
    "make_rotation",
    "apply_rotation",
    "kurtosis",
    "verify_gaussianization",
    "fit_zca_whitening",
    "apply_whitening",
    # alignment
    "identity_projection",
    "orthogonal_procrustes",
    "orthogonal_procrustes_randomized",
    "ridge_projection",
    "cca_projection",
    "reduced_rank_regression",
    "fit_alignment",
    "alignment_quality",
    # quantization
    "GaussianQuantizer",
    "lloyd_max_gaussian",
    # top-level
    "RotAlignKVTranslator",
    "TranslatorConfig",
]
