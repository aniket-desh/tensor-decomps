"""
cp decomposition module.
provides optimizers for canonical polyadic (cp) tensor decomposition.
"""

from .standard_ALS import CP_DTALS_Optimizer, CP_PPALS_Optimizer
from .NLS import CP_fastNLS_Optimizer
from .mahalanobis import CP_AMDM_Optimizer, CP_AMDM_MLE_Optimizer
from .common_kernels import (
    cp_reconstruct,
    mahalanobis_norm,
    get_residual,
    get_residual3,
    get_residual_sp,
    get_residual_sp3,
    compute_lin_sysN,
    compute_lin_sys,
    compute_condition_number,
    solve_sys,
    flatten_Tensor,
    reshape_into_matrices,
    compute_number_of_variables,
    equilibrate,
    normalise,
)

__all__ = [
    # optimizers
    'CP_DTALS_Optimizer',
    'CP_PPALS_Optimizer', 
    'CP_fastNLS_Optimizer',
    'CP_AMDM_Optimizer',
    'CP_AMDM_MLE_Optimizer',
    # tensor operations
    'cp_reconstruct',
    'mahalanobis_norm',
    # kernels
    'get_residual',
    'get_residual3',
    'get_residual_sp',
    'get_residual_sp3',
    'compute_lin_sysN',
    'compute_lin_sys',
    'compute_condition_number',
    'solve_sys',
    'flatten_Tensor',
    'reshape_into_matrices',
    'compute_number_of_variables',
    'equilibrate',
    'normalise',
]

