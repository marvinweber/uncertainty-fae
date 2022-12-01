from clavicle_ct.litmodel.base import LitClavicle, LitClavicleVarianceNet
from clavicle_ct.litmodel.deep_ensemble import (
    LitClavicleDeepEnsemble,
    LitClavicleVarianceNetDeepEnsemble,
)
from clavicle_ct.litmodel.dropout import LitClavicleMCDropout, LitClavicleVarianceNetMCDropout
from clavicle_ct.litmodel.laplace import LitClavicleLaplace
from clavicle_ct.litmodel.swag import LitClavicleSWAG

__all__ = [
    # Base LitModels
    "LitClavicle",
    "LitClavicleVarianceNet",
    # Dropout LitModels
    "LitClavicleMCDropout",
    "LitClavicleVarianceNetMCDropout",
    # Deep Ensemble LitModels
    "LitClavicleDeepEnsemble",
    "LitClavicleVarianceNetDeepEnsemble",
    # Laplace Approximation LitModels
    "LitClavicleLaplace",
    # SWAG LitModels
    "LitClavicleSWAG",
]
