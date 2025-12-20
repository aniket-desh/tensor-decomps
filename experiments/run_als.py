"""
alternating least squares (als) optimization for cp decomposition.
supports standard als (dt) and pairwise perturbation (pp) methods.
"""

import numpy as np
import time
import csv
from pathlib import Path
from os.path import dirname, join

# centralized imports from tensor_decomposition
import tensor_decomposition as td
from tensor_decomposition import (
    CP_DTALS_Optimizer,
    CP_PPALS_Optimizer,
    get_residual,
    get_residual_sp,
    compute_condition_number,
    save_decomposition_results,
    get_file_prefix,
)
from tensor_decomposition.utils import arg_defs

PARENT_DIR = dirname(__file__)
RESULTS_DIR = join(PARENT_DIR, 'results')


def cp_als(
    tenpy,
    tensor_true,
    initial_factors,
    tensor,
    sparsity_pattern,
    args,
    cov_empirical,
    cov_pinv_empirical,
    m_empirical_pinv,
    csv_file=None,
    regularization=None,
    method='DT',
    res_calc_freq=1,
    tol=1e-05
):
    """
    run cp decomposition using alternating least squares.
    
    performs multiple runs with random initializations and tracks
    residuals, mahalanobis norms, and other metrics.
    """
    final_residuals = []
    all_residuals = []
    all_norm_mahalanobis_empirical = []
    
    for run in range(args.num_runs):
        # random initialization for each run
        factors = [np.random.rand(tensor.shape[i], args.R) for i in range(tensor.ndim)]
        print(f"[trial {run+1:02d}/{args.num_runs:02d}] starting als optimization")
        
        residuals = []
        norm_mahalanobis_emp = []
        flag_dt = True
        
        if csv_file is not None:
            csv_writer = csv.writer(
                csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL
            )
            
        if regularization is None:
            regularization = 0
        
        norm_t = tenpy.vecnorm(tensor)
        time_all = 0.0
        
        if args is None:
            optimizer = CP_DTALS_Optimizer(tenpy, tensor, factors, args)
        else:
            optimizer_list = {
                'DT': CP_DTALS_Optimizer(tenpy, tensor, factors, args),
                'PP': CP_PPALS_Optimizer(tenpy, tensor, factors, args),
            }
            optimizer = optimizer_list[method]
        
        fitness_old = 0
        for i in range(args.num_iter):
            # compute residual at specified frequency
            if i % res_calc_freq == 0 or i == args.num_iter - 1 or not flag_dt:
                if args.fast_residual and i != 0:
                    res = optimizer.compute_fast_residual()
                else:
                    if args.sp and sparsity_pattern is not None:
                        res = get_residual_sp(tenpy, sparsity_pattern, tensor, factors)
                    else:
                        res = get_residual(tenpy, tensor, factors)
                
                residuals.append(res)
                fitness = 1 - res / norm_t
                
                # compute mahalanobis norm
                t_reconstructed = tenpy.zeros(tensor.shape)
                t_reconstructed = t_reconstructed + tenpy.einsum('ir,jr,kr->ijk', *factors)
                diff = tensor_true - t_reconstructed
                norm_mahalanobis_empirical = np.einsum(
                    'ip,jq,kr,ijk,pqr->', *m_empirical_pinv, diff, diff
                )
                norm_mahalanobis_emp.append(norm_mahalanobis_empirical)
                
                # optional condition number calculation
                if args.calc_cond and args.R < 15 and tenpy.name() == 'numpy':
                    cond = compute_condition_number(tenpy, factors)
                    if tenpy.is_master_proc():
                        print(f"[info] iter={i} | residual={res:.2e} | fitness={fitness:.4f}")
                        if csv_file is not None:
                            csv_writer.writerow([
                                i, time_all, res, fitness, flag_dt, cond,
                                norm_mahalanobis_empirical
                            ])
                            csv_file.flush()
                else:
                    if tenpy.is_master_proc():
                        print(f"[info] iter={i} | residual={res:.2e} | fitness={fitness:.4f}")
                        if csv_file is not None:
                            csv_writer.writerow([
                                i, time_all, res, fitness, flag_dt,
                                norm_mahalanobis_empirical
                            ])
                            csv_file.flush()
            
            # check convergence
            if res < tol:
                print(f'[info] converged in {i} iterations')
                break
            
            # optimization step
            t0 = time.time()
            if method == 'PP':
                factors, pp_restart = optimizer.step(regularization)
                flag_dt = not pp_restart
            else:
                factors = optimizer.step(regularization)
            t1 = time.time()
            
            tenpy.printf(f"[info] iter={i} | sweep time={t1-t0:.2f}s")
            time_all += t1 - t0
            fitness_old = fitness
        
        # store results for this run
        final_residuals.append(residuals[-1])
        all_residuals.append(residuals)
        all_norm_mahalanobis_empirical.append(norm_mahalanobis_emp)
        
        tenpy.printf(f"[summary] {method} method took {time_all:.2f}s overall")
        
        if args.save_tensor:
            folderpath = join(RESULTS_DIR, get_file_prefix(args))
            save_decomposition_results(tensor, factors, tenpy, folderpath)
    
    # compute statistics across runs
    best_run_index = np.argmin(final_residuals)
    best_run_residual = all_residuals[best_run_index]
    best_run_norm_mahalanobis_empirical = all_norm_mahalanobis_empirical[best_run_index]
    final_residuals = np.sort(final_residuals)[::-1]
    
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
    
    return (
        best_run_residual,
        best_run_norm_mahalanobis_empirical,
        final_residuals,
        mean_residuals,
        std_residuals
    )
