"""
tensor decomposition library.

a python library for efficient tensor decomposition, featuring multiple
optimization methods for cp (canonical polyadic) and tucker decompositions.

quick start:
    >>> import tensor_decomposition as td
    >>> 
    >>> # get a backend
    >>> tenpy = td.get_backend('numpy')
    >>> 
    >>> # create a random tensor
    >>> tensor = tenpy.random((10, 10, 10))
    >>> 
    >>> # initialize factor matrices
    >>> factors = [tenpy.random((10, 5)) for _ in range(3)]
    >>> 
    >>> # create an optimizer
    >>> opt = td.CP_DTALS_Optimizer(tenpy, tensor, factors, args)
    >>> 
    >>> # run optimization
    >>> for i in range(100):
    ...     factors = opt.step(regularization=1e-6)
"""

# version
try:
    from pathlib import Path
    _version_file = Path(__file__).parent / 'VERSION'
    __version__ = _version_file.read_text().strip() if _version_file.exists() else '0.1.0'
except:
    __version__ = '0.1.0'

# backend
from .backend import get_backend

# cp decomposition optimizers
from .CPD import (
    CP_DTALS_Optimizer,
    CP_PPALS_Optimizer,
    CP_fastNLS_Optimizer,
    CP_AMDM_Optimizer,
)

# cp decomposition kernels
from .CPD import (
    get_residual,
    get_residual3,
    get_residual_sp,
    compute_condition_number,
)

# als base classes
from .als import DTALS_base, PPALS_base

# utilities
from .utils import (
    generate_initial_guess,
    generate_tensor,
    save_decomposition_results,
    plot_convergence,
    plot_comparison_results,
)

# argument parsing utilities
from .utils import (
    add_general_arguments,
    add_nls_arguments,
    add_amdm_arguments,
    add_sparse_arguments,
    add_pp_arguments,
    add_col_arguments,
    get_file_prefix,
)

# submodules for direct access
from . import CPD
from . import Tucker
from . import tensors
from . import backend
from . import utils

__all__ = [
    # version
    '__version__',
    # backend
    'get_backend',
    # optimizers
    'CP_DTALS_Optimizer',
    'CP_PPALS_Optimizer',
    'CP_fastNLS_Optimizer',
    'CP_AMDM_Optimizer',
    # base classes
    'DTALS_base',
    'PPALS_base',
    # kernels
    'get_residual',
    'get_residual3',
    'get_residual_sp',
    'compute_condition_number',
    # utilities
    'generate_initial_guess',
    'generate_tensor',
    'save_decomposition_results',
    'plot_convergence',
    'plot_comparison_results',
    # argument parsing
    'add_general_arguments',
    'add_nls_arguments',
    'add_amdm_arguments',
    'add_sparse_arguments',
    'add_pp_arguments',
    'add_col_arguments',
    'get_file_prefix',
    # submodules
    'CPD',
    'Tucker',
    'tensors',
    'backend',
    'utils',
]
