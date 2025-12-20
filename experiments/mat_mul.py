"""
matrix multiplication tensor decomposition experiments.
tests als, gauss-newton, and hybrid methods on the matrix multiplication tensor.
"""

import argparse
import time
import numpy as np
import csv
from pathlib import Path
from os.path import dirname, join

# centralized imports from tensor_decomposition
import tensor_decomposition as td
from tensor_decomposition import (
    CP_DTALS_Optimizer,
    CP_fastNLS_Optimizer,
    get_residual,
)

PARENT_DIR = dirname(__file__)
RESULTS_DIR = join(PARENT_DIR, 'results')


def init_matmul_tensor(tenpy, m1, m2, m3, seed=1):
    """
    construct the matrix multiplication tensor.
    
    the tensor encodes the bilinear map for multiplying an m1 x m2 matrix
    with an m2 x m3 matrix to produce an m1 x m3 matrix.
    """
    i1 = tenpy.speye(m2)
    i2 = tenpy.speye(m1)
    i3 = tenpy.speye(m3)
    tensor = tenpy.einsum("lm,ik,nj->ijklmn", i1, i2, i3)
    tensor = tensor.reshape((m1 * m3, m1 * m2, m2 * m3))
    return [tensor, tensor]


def run_matmul_experiment(
    tenpy, m1, m2, m3, rank, seed_start, tries,
    tol_init, tol_fin, method, csv_file, csv_writer
):
    """
    run matrix multiplication tensor decomposition experiment.
    
    tests convergence of different methods across multiple random initializations.
    """
    [tensor, sparsity_pattern] = init_matmul_tensor(tenpy, m1, m2, m3, seed=1)
    
    conv_count = 0
    total_time = 0
    total_iters = 0
    
    sizes = [m1 * m3, m1 * m2, m2 * m3]
    
    print(f"[info] running {method} method with rank={rank}")
    
    # create mock args for optimizer
    class MockArgs:
        def __init__(self):
            self.maxiter = 3 * np.max(sizes) * rank
            self.cg_tol = 1e-03
            self.num = 0
            self.diag = 0
            self.arm = 0
            self.c = 0.5
            self.tau = 0.5
            self.arm_iters = 10
            self.sp = 0
            self.fast_residual = 0
    
    mock_args = MockArgs()
    
    if method == 'ALS':
        total_start = time.time()
        
        for trial in range(tries):
            start = time.time()
            seed = 1301 * seed_start + 131 * trial
            np.random.seed(seed)
            
            # random initialization
            factors = [np.random.randn(s, rank) for s in sizes]
            
            # initialization phase
            num_iter = 250
            regu = 1e-02
            opt = CP_DTALS_Optimizer(tenpy, tensor, factors, mock_args)
            
            for i in range(num_iter):
                factors = opt.step(regu)
                res = get_residual(tenpy, tensor, factors)
                if res < tol_init:
                    tenpy.printf(f"[info] init converged in {i} iterations")
                    break
            
            # main optimization phase
            opt = CP_DTALS_Optimizer(tenpy, tensor, factors, mock_args)
            num_iter = 10000
            regu = 1e-02
            
            for i in range(num_iter):
                factors = opt.step(regu)
                res = get_residual(tenpy, tensor, factors)
                
                if res < tol_fin:
                    tenpy.printf(f'[info] converged in {i} iterations | residual={res:.2e}')
                    conv_count += 1
                    total_iters += i
                    total_time += time.time() - start
                    break
                
                if regu > 1e-10:
                    regu = regu / 2
        
        total_end = time.time()
        print(f"[summary] trials converged: {conv_count}/{tries}")
        
    elif method == 'GN':
        total_start = time.time()
        cg_iters = 0
        
        for trial in range(tries):
            start = time.time()
            seed = 1301 * seed_start + 131 * trial
            np.random.seed(seed)
            
            factors = [np.random.randn(s, rank) for s in sizes]
            
            # initialization phase
            mock_args.diag = 1
            num_iter = 100
            regu = 1e-02
            opt = CP_fastNLS_Optimizer(tenpy, tensor, factors, mock_args)
            
            for i in range(num_iter):
                factors = opt.step(regu)
                res = get_residual(tenpy, tensor, factors)
                if res < tol_init:
                    tenpy.printf(f"[info] init converged in {i} iterations")
                    break
            
            # main optimization with adaptive regularization
            mock_args.diag = 0
            opt = CP_fastNLS_Optimizer(tenpy, tensor, factors, mock_args)
            
            num_iter = 300
            regu = 1e-03
            lower = 1e-07
            upper = 1e-03
            fact = 2
            decrease = True
            increase = False
            
            for i in range(num_iter):
                factors = opt.step(regu)
                res = get_residual(tenpy, tensor, factors)
                
                if res < tol_fin:
                    tenpy.printf(f'[info] converged in {i} iterations | residual={res:.2e}')
                    conv_count += 1
                    total_time += time.time() - start
                    break
                
                # adaptive regularization
                if regu < lower:
                    increase = True
                    decrease = False
                if regu > upper:
                    decrease = True
                    increase = False
                if increase:
                    regu = regu * fact
                elif decrease:
                    regu = regu / fact
        
        total_end = time.time()
        print(f"[summary] trials converged: {conv_count}/{tries}")
        
    elif method == 'HYB':
        total_start = time.time()
        cg_iters = 0
        
        for trial in range(tries):
            start = time.time()
            seed = 1301 * seed_start + 131 * trial
            np.random.seed(seed)
            
            factors = [np.random.randn(s, rank) for s in sizes]
            
            # als initialization phase
            num_iter = 150
            regu = 1e-02
            opt = CP_DTALS_Optimizer(tenpy, tensor, factors, mock_args)
            
            for i in range(num_iter):
                factors = opt.step(regu)
                res = get_residual(tenpy, tensor, factors)
                if res < tol_init:
                    tenpy.printf(f"[info] init converged in {i} iterations")
                    break
            
            # gauss-newton refinement with armijo line search
            mock_args.arm = 1
            mock_args.cg_tol = 0.5
            opt = CP_fastNLS_Optimizer(tenpy, tensor, factors, mock_args)
            
            num_iter = 300
            regu = 1e-02
            lower = 1e-07
            upper = 1e-03
            fact = 2
            decrease = True
            increase = False
            
            for i in range(num_iter):
                factors = opt.step(regu)
                res = get_residual(tenpy, tensor, factors)
                
                if res < tol_fin:
                    tenpy.printf(f'[info] converged in {i} iterations | residual={res:.2e}')
                    conv_count += 1
                    total_time += time.time() - start
                    break
                
                # adaptive regularization
                if regu < lower:
                    increase = True
                    decrease = False
                if regu > upper:
                    decrease = True
                    increase = False
                if increase:
                    regu = regu * fact
                elif decrease:
                    regu = regu / fact
        
        total_end = time.time()
        print(f"[summary] trials converged: {conv_count}/{tries}")
    
    # log results
    if tenpy.is_master_proc() and csv_file is not None:
        avg_iters = total_iters / tries if tries > 0 else 0
        avg_time = total_time / conv_count if conv_count > 0 else 0
        csv_writer.writerow([
            m1, m2, m3, rank, method, tries, conv_count,
            avg_iters, total_end - total_start, avg_time, seed_start
        ])
        csv_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tlib', default="numpy", metavar='string',
        choices=['ctf', 'numpy'],
        help='tensor library to use (default: numpy)'
    )
    parser.add_argument(
        '--R', type=int, default=4, metavar="int",
        help="rank for matrix multiplication tensor (default: 4)"
    )
    parser.add_argument(
        '--m1', type=int, default=3, metavar="int",
        help="first dimension (default: 3)"
    )
    parser.add_argument(
        '--m2', type=int, default=3, metavar="int",
        help="second dimension (default: 3)"
    )
    parser.add_argument(
        '--m3', type=int, default=3, metavar="int",
        help="third dimension (default: 3)"
    )
    parser.add_argument(
        '--tol-init', type=float, default=0.01, metavar="float",
        help="initialization tolerance (default: 0.01)"
    )
    parser.add_argument(
        '--tol-fin', type=float, default=1e-08, metavar="float",
        help="final tolerance (default: 1e-08)"
    )
    parser.add_argument(
        '--method', default='HYB', metavar='string',
        choices=['GN', 'ALS', 'HYB'],
        help='optimization method (default: HYB)'
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar="int",
        help="random seed"
    )
    parser.add_argument(
        '--tries', type=int, default=5, metavar="int",
        help="number of trials (default: 5)"
    )
    
    args, _ = parser.parse_known_args()
    
    # set up csv logging
    csv_path = join(RESULTS_DIR, 'matmul.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    tlib = args.tlib
    rank = args.R
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    tries = args.tries
    tol_init = args.tol_init
    tol_fin = args.tol_fin
    method = args.method
    seed = args.seed
    
    # load tensor backend
    tenpy = td.get_backend(tlib)
    
    if tenpy.is_master_proc():
        # print configuration
        print("[info] experiment configuration:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        
        # initialize csv header
        if is_new_log:
            csv_writer.writerow([
                'dim1', 'dim2', 'dim3', 'rank', 'method', 'trials', 'converged',
                'avg_iterations', 'total_time', 'avg_time_converged', 'seed'
            ])

    run_matmul_experiment(
        tenpy, m1, m2, m3, rank, seed, tries,
        tol_init, tol_fin, method, csv_file, csv_writer
    )
    
    print(f"[done] results saved -> {csv_path}")
