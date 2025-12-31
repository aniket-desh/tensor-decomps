"""
alternating mahalanobis distance minimization (amdm) for cp decomposition.
optimizes using the mahalanobis norm instead of the standard frobenius norm.
"""

import numpy as np
import time
import csv
from pathlib import Path
from os.path import dirname, join

# centralized imports from tensor_decomposition
import tensor_decomposition as td
from tensor_decomposition import (
    CP_AMDM_Optimizer,
    CP_AMDM_MLE_Optimizer,
    get_residual,
    get_residual_sp,
    compute_condition_number,
    save_decomposition_results,
    get_file_prefix,
    cp_reconstruct,
    mahalanobis_norm,
    factor_match_score,
)
from tensor_decomposition.utils import arg_defs

PARENT_DIR = dirname(__file__)
RESULTS_DIR = join(PARENT_DIR, 'results')


# csv header for experiment logging
CSV_HEADER = [
    'method', 'seed', 'trial_id', 'iter', 'time',
    'order', 's', 'R', 'epsilon', 'k', 'alpha',
    'residual', 'fitness', 'norm_mahal', 'fms', 'cond'
]


def cp_mahalanobis(
    tenpy,
    tensor_true,
    initial_factors,
    tensor,
    sparsity_pattern,
    args,
    factors_true=None,
    m_empirical_pinv=None,
    csv_file=None,
    regularization=None,
    res_calc_freq=1,
    use_mle_optimizer=False
):
    """
    run cp decomposition using alternating mahalanobis distance minimization.
    
    performs multiple runs and tracks residuals, mahalanobis norms, factor match
    scores, and other metrics.
    
    args:
        tenpy: tensor backend (numpy or ctf)
        tensor_true: ground-truth tensor (for computing mahalanobis norm)
        initial_factors: initial factor matrices (used if provided, else random)
        tensor: input tensor to decompose
        sparsity_pattern: sparsity pattern (for sparse tensors)
        args: argument namespace with optimization parameters
        factors_true: ground-truth factor matrices (for computing fms)
        m_empirical_pinv: metric factors for mahalanobis norm
        csv_file: file handle for logging (optional)
        regularization: regularization parameter
        res_calc_freq: frequency of residual calculation
        use_mle_optimizer: if True, use CP_AMDM_MLE_Optimizer with known metric
        
    returns:
        dict with keys: best_residuals, best_mahalanobis, best_fms, best_factors,
                        final_residuals, final_fms, mean_residuals, std_residuals
    """
    final_residuals = []
    final_fms = []
    all_residuals = []
    all_norm_mahalanobis_empirical = []
    all_fms = []
    final_factors_list = []
    
    # get rank - use R_app if available, otherwise R
    rank = getattr(args, 'R_app', args.R)
    
    # extract parameters for logging
    seed = getattr(args, 'seed', 0)
    order = getattr(args, 'order', tensor.ndim)
    s = getattr(args, 's', tensor.shape[0])
    R = getattr(args, 'R', rank)
    epsilon = getattr(args, 'epsilon', 0)
    k_param = getattr(args, 'k', 0)
    alpha = getattr(args, 'alpha', 1.0)
    
    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL
        )
    
    for run in range(args.num_runs):
        # use provided initial_factors for first run, or generate random for each run
        # this ensures reproducibility while allowing fair comparison
        if initial_factors is not None and run == 0:
            factors = [f.copy() for f in initial_factors]
            print(f"[trial {run+1:02d}/{args.num_runs:02d}] starting amdm optimization (provided init)")
        else:
            # use backend's random generator for reproducibility with CTF
            tenpy.seed(args.seed * 1001 + run)
            factors = [tenpy.random((tensor.shape[i], rank)) for i in range(tensor.ndim)]
            print(f"[trial {run+1:02d}/{args.num_runs:02d}] starting amdm optimization (random init)")

        residuals = []
        norm_mahalanobis_emp = []
        fms_history = []
            
        if regularization is None:
            regularization = 0
        
        time_all = 0.0
        norm_t = tenpy.vecnorm(tensor)
        
        # choose optimizer based on whether we have known metric factors
        if use_mle_optimizer and m_empirical_pinv is not None:
            method = 'AMDM-MLE'
            optimizer = CP_AMDM_MLE_Optimizer(tenpy, tensor, factors, m_empirical_pinv, args)
        else:
            method = 'AMDM'
            optimizer = CP_AMDM_Optimizer(tenpy, tensor, factors, args)
        
        for iter_idx in range(args.num_iter):
            # compute residual at specified frequency
            if iter_idx % res_calc_freq == 0 or iter_idx == args.num_iter - 1:
                if args.sp and sparsity_pattern is not None:
                    res = get_residual_sp(tenpy, sparsity_pattern, tensor, factors)
                else:
                    res = get_residual(tenpy, tensor, factors)
                
                residuals.append(res)
                fitness = 1 - res / norm_t
                
                # compute mahalanobis norm using generalized function
                if m_empirical_pinv is not None:
                    t_reconstructed = cp_reconstruct(tenpy, factors)
                    diff = tensor_true - t_reconstructed
                    norm_mahalanobis_empirical = mahalanobis_norm(tenpy, diff, m_empirical_pinv)
                else:
                    norm_mahalanobis_empirical = 0.0
                norm_mahalanobis_emp.append(norm_mahalanobis_empirical)
                
                # compute factor match score if ground-truth factors available
                if factors_true is not None:
                    fms = factor_match_score(factors_true, factors)
                else:
                    fms = 0.0
                fms_history.append(fms)
                
                # optional condition number calculation
                if args.calc_cond and args.R < 15 and tenpy.name() == 'numpy':
                    cond = compute_condition_number(tenpy, factors)
                else:
                    cond = 0.0
                
                if tenpy.is_master_proc():
                    print(f"[info] iter={iter_idx} | residual={res:.2e} | fitness={fitness:.4f} | fms={fms:.4f}")
                    if csv_file is not None:
                        csv_writer.writerow([
                            method, seed, run, iter_idx, time_all,
                            order, s, R, epsilon, k_param, alpha,
                            res, fitness, norm_mahalanobis_empirical, fms, cond
                        ])
                        csv_file.flush()
            
            # optimization step
            t0 = time.time()
            factors = optimizer.step(regularization)
            
            # optional value reduction for hybrid algorithm (only for standard AMDM)
            if hasattr(args, 'reduce_val') and args.reduce_val:
                if hasattr(optimizer, 'reduce_vals') and iter_idx > 0 and iter_idx % args.reduce_val_freq == 0:
                    optimizer.reduce_vals()
            
            t1 = time.time()
            tenpy.printf(f"[info] iter={iter_idx} | sweep time={t1-t0:.2f}s")
            time_all += t1 - t0
            
            # check convergence criteria
            if res < args.tol:
                tenpy.printf(f'[info] converged due to residual tol in {iter_idx} iterations')
                break
            
            if hasattr(args, 'fit') and fitness > args.fit:
                tenpy.printf(f'[info] converged due to fitness tol in {iter_idx} iterations')
                break
        
        # store results for this run
        final_residuals.append(residuals[-1])
        final_fms.append(fms_history[-1] if fms_history else 0.0)
        all_residuals.append(residuals)
        all_norm_mahalanobis_empirical.append(norm_mahalanobis_emp)
        all_fms.append(fms_history)
        final_factors_list.append([f.copy() for f in factors])
        
        tenpy.printf(f"[summary] {method} method took {time_all:.2f}s overall")
        
        if args.save_tensor:
            folderpath = join(RESULTS_DIR, get_file_prefix(args))
            save_decomposition_results(tensor, factors, tenpy, folderpath)
    
    # compute statistics across runs
    best_run_index = np.argmin(final_residuals)
    best_run_residual = all_residuals[best_run_index]
    best_run_norm_mahalanobis_empirical = all_norm_mahalanobis_empirical[best_run_index]
    best_run_fms = all_fms[best_run_index] if all_fms else []
    best_factors = final_factors_list[best_run_index]
    final_residuals_sorted = np.sort(final_residuals)[::-1]
    
    min_length = min(len(residuals) for residuals in all_residuals)
    truncated_residuals = [residuals[-min_length:] for residuals in all_residuals]
    truncated_norm_mahalanobis_empirical = [
        rr[-min_length:] for rr in all_norm_mahalanobis_empirical
    ]
    
    truncated_residuals = np.array(truncated_residuals)
    truncated_norm_mahalanobis_empirical = np.array(truncated_norm_mahalanobis_empirical)
    mean_residuals = np.mean(truncated_residuals, axis=0)
    mean_norm_mahalanobis_empirical = np.mean(truncated_norm_mahalanobis_empirical, axis=0)
    std_residuals = np.std(truncated_residuals, axis=0)
    
    return {
        'best_residuals': best_run_residual,
        'best_mahalanobis': best_run_norm_mahalanobis_empirical,
        'best_fms': best_run_fms,
        'best_factors': best_factors,
        'final_residuals': final_residuals_sorted,
        'final_fms': final_fms,
        'mean_residuals': mean_residuals,
        'std_residuals': std_residuals,
    }
