"""
utility module for tensor decomposition.
provides argument parsing, tensor generation, plotting, metrics, and helper functions.
"""

from .generators import generate_initial_guess, generate_tensor
from .plotting import plot_convergence, plot_comparison_results, lighten_color
from .utils import save_decomposition_results
from .metrics import (
    factor_match_score,
    aligned_factor_error,
    congruence_coefficient,
    relative_factor_error,
    cosine_similarity,
)
from .arg_defs import (
    add_general_arguments,
    add_pp_arguments,
    add_col_arguments,
    add_lrdt_arguments,
    add_sparse_arguments,
    add_amdm_arguments,
    add_nls_arguments,
    add_probability_arguments,
    get_file_prefix,
    get_prob_file_prefix,
)

__all__ = [
    # generators
    'generate_initial_guess',
    'generate_tensor',
    # plotting
    'plot_convergence',
    'plot_comparison_results',
    'lighten_color',
    # io
    'save_decomposition_results',
    # metrics
    'factor_match_score',
    'aligned_factor_error',
    'congruence_coefficient',
    'relative_factor_error',
    'cosine_similarity',
    # argument parsing
    'add_general_arguments',
    'add_pp_arguments',
    'add_col_arguments',
    'add_lrdt_arguments',
    'add_sparse_arguments',
    'add_amdm_arguments',
    'add_nls_arguments',
    'add_probability_arguments',
    'get_file_prefix',
    'get_prob_file_prefix',
]

