"""
nonlinear least squares (nls) optimization for cp decomposition.
uses gauss-newton method with preconditioned conjugate gradient.
"""

import numpy as np
import time
import argparse
import csv
from pathlib import Path
from os.path import dirname, join

# centralized imports from tensor_decomposition
import tensor_decomposition as td
from tensor_decomposition import (
    CP_fastNLS_Optimizer,
    get_residual,
    get_residual_sp,
    save_decomposition_results,
    get_file_prefix,
)
from tensor_decomposition.tensors import synthetic_tensors, real_tensors
from tensor_decomposition.utils import arg_defs

PARENT_DIR = dirname(__file__)
RESULTS_DIR = join(PARENT_DIR, 'results')


def cp_nls(
    tenpy,
    factors,
    tensor,
    sparsity_pattern,
    num_iter,
    csv_file=None,
    regularization=None,
    method='NLS',
    args=None,
    res_calc_freq=1
):
    """
    run cp decomposition using fast nonlinear least squares.
    
    solves the gauss-newton equations using preconditioned conjugate gradient
    for efficient optimization of all factor matrices simultaneously.
    """
    if csv_file is not None:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL
        )

    if regularization is None:
        regularization = 0
        
    if args.varying:
        decrease = True
        increase = False
    
    iters = 0
    count = 0
    
    norm_t = tenpy.vecnorm(tensor)
    
    if args.maxiter == 0:
        args.maxiter = sum(tensor.shape) * args.R
    
    time_all = 0.0
    
    # initialize optimizer
    if method == 'DT':
        method = 'NLS'
        optimizer = CP_fastNLS_Optimizer(tenpy, tensor, factors, args)
    else:
        optimizer_list = {
            'NLS': CP_fastNLS_Optimizer(tenpy, tensor, factors, args)
        }
        optimizer = optimizer_list[method]

    fitness_old = 0
    prev_res = np.finfo(np.float32).max
    
    for i in range(num_iter):
        # compute residual at specified frequency
        if i % res_calc_freq == 0 or i == num_iter - 1:
            if args.sp and sparsity_pattern is not None:
                res = get_residual_sp(tenpy, sparsity_pattern, tensor, factors)
            else:
                res = get_residual(tenpy, tensor, factors)
            fitness = 1 - res / norm_t

            if tenpy.is_master_proc():
                print(f"[info] iter={i} | residual={res:.2e} | fitness={fitness:.4f}")
                if csv_file is not None:
                    if method == 'NLS':
                        csv_writer.writerow([iters, time_all, res, fitness])
                    else:
                        csv_writer.writerow([i, time_all, res, fitness])
                    csv_file.flush()
        
        # check convergence
        if res < args.nls_tol:
            tenpy.printf(f'[info] converged due to residual tol in {i} iterations')
            break
        
        # optimization step
        t0 = time.time()
        factors = optimizer.step(regularization)
        count += 1
        t1 = time.time()
        
        tenpy.printf(f"[info] iter={i} | sweep time={t1-t0:.2f}s")
        time_all += t1 - t0
        
        # check gradient convergence for nls
        if method == 'NLS':
            if optimizer.g_norm < args.grad_tol:
                tenpy.printf(f'[info] converged due to gradient tol in {i} iterations')
                break
        
        # adaptive regularization
        if args.varying:
            if regularization < args.lower:
                increase = True
                decrease = False
            if regularization > args.upper:
                decrease = True
                increase = False
            if increase:
                regularization = regularization * args.varying_fact
            elif decrease:
                regularization = regularization / args.varying_fact
    
    tenpy.printf(f"[summary] {method} method took {time_all:.2f}s overall")
    
    if args.save_tensor:
        folderpath = join(RESULTS_DIR, get_file_prefix(args))
        save_decomposition_results(tensor, factors, tenpy, folderpath)

    return factors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_lrdt_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_nls_arguments(parser)
    arg_defs.add_col_arguments(parser)
    args, _ = parser.parse_known_args()

    # set up csv logging
    csv_path = join(RESULTS_DIR, get_file_prefix(args) + '.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    size = args.s
    order = args.order
    rank = args.R
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tensor_type = args.tensor
    tlib = args.tlib

    # load tensor backend
    tenpy = td.get_backend(tlib)
    
    if tlib == "ctf":
        import ctf
        tepoch = ctf.timer_epoch("NLS")
        tepoch.begin()

    if tenpy.is_master_proc():
        # print configuration
        print("[info] experiment configuration:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        # initialize csv header
        if is_new_log:
            csv_writer.writerow(['iterations', 'time', 'residual', 'fitness'])

    tenpy.seed(args.seed)

    # generate or load tensor
    sparsity_pattern = None
    if args.load_tensor != '':
        tensor = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
    elif tensor_type == "random":
        tenpy.printf("[info] testing random tensor")
        [tensor, sparsity_pattern] = synthetic_tensors.rand(
            tenpy, order, size, rank, sp_frac, args.seed
        )
    elif tensor_type == "random_col":
        [tensor, sparsity_pattern] = synthetic_tensors.collinearity_tensor(
            tenpy, size, order, rank, args.col, args.seed
        )
    elif tensor_type == "amino":
        tensor = real_tensors.amino_acids(tenpy)
    elif tensor_type == "negrandom":
        tenpy.printf("[info] testing random tensor with negative entries")
        [tensor, sparsity_pattern] = synthetic_tensors.neg_rand(
            tenpy, order, size, rank, sp_frac, args.seed
        )
    elif tensor_type == "randn":
        tenpy.printf("[info] testing random tensor with normally distributed entries")
        [tensor, sparsity_pattern] = synthetic_tensors.randn(
            tenpy, order, size, rank, sp_frac, args.seed
        )
    else:
        raise ValueError(f"[error] unknown tensor type: {tensor_type}")
        
    tenpy.printf(f"[info] input tensor shape: {tensor.shape}")

    regularization = args.regularization

    # initialize factor matrices
    factors = []
    if args.load_tensor != '':
        for i in range(tensor.ndim):
            factors.append(tenpy.load_tensor_from_file(args.load_tensor + 'mat' + str(i) + '.npy'))
    else:
        if args.decomposition == "CP":
            for i in range(tensor.ndim):
                factors.append(tenpy.random((tensor.shape[i], rank)))

    if args.decomposition == 'Tucker':
        raise ValueError("[error] tucker decomposition is not supported via gauss-newton")

    cp_nls(
        tenpy, factors, tensor, sparsity_pattern, num_iter,
        csv_file, regularization, args.method, args, args.res_calc_freq
    )
    
    print(f"[done] results saved -> {csv_path}")
