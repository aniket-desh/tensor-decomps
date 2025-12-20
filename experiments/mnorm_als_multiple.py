"""
multiple tensor experiment comparing mahalanobis and als methods.
runs experiments across multiple randomly generated tensors.
"""

import numpy as np
import sys
import time
import os
import argparse
import csv
from pathlib import Path
from os.path import dirname, join

import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import tensor_decomposition.utils.arg_defs as arg_defs
from tensor_decomposition.CPD.common_kernels import get_residual, get_residual_sp, compute_condition_number
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.mahalanobis import CP_AMDM_Optimizer
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer
from run_als import cp_als
from run_mahalanobis import cp_mahalanobis

PARENT_DIR = dirname(__file__)
RESULTS_DIR = join(PARENT_DIR, 'results')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_amdm_arguments(parser)
    args, _ = parser.parse_known_args()

    size = args.s
    order = args.order
    rank = args.R
    
    if args.R_app is None:
        rank_app = args.R
    else:
        rank_app = args.R_app

    if args.num_vals is None:
        args.num_vals = rank_app
    
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tensor_type = args.tensor
    tlib = args.tlib

    # load tensor backend
    if tlib == "numpy":
        import tensor_decomposition.backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import tensor_decomposition.backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin()

    # generate or load tensor
    sparsity_pattern = None
    if args.load_tensor != '':
        tensor = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
    elif tensor_type == "random":
        tenpy.printf("[info] testing random tensor")
        [tensor, sparsity_pattern] = synthetic_tensors.rand(
            tenpy, order, size, rank, sp_frac, np.random.randint(100)
        )
    elif tensor_type == "MGH":
        tensor = tenpy.load_tensor_from_file("MGH-16.npy")
        tensor = tensor.reshape(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2], tensor.shape[3], tensor.shape[4]
        )
    elif tensor_type == "SLEEP":
        tensor = tenpy.load_tensor_from_file("SLEEP-16.npy")
        tensor = tensor.reshape(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2], tensor.shape[3], tensor.shape[4]
        )
    elif tensor_type == "random_col":
        [tensor, sparsity_pattern] = synthetic_tensors.collinearity_tensor(
            tenpy, size, order, rank, args.col, np.random.randint(100)
        )
    elif tensor_type == "scf":
        tensor = np.load('scf_tensor.npy')
    elif tensor_type == "amino":
        tensor = real_tensors.amino_acids(tenpy)

    tenpy.printf(f"[info] input tensor shape: {tensor.shape}")

    num_tensors = 5
    
    for tensor_idx in range(num_tensors):
        print(f"\n[info] processing tensor {tensor_idx + 1}/{num_tensors}")
        
        regularization = args.regularization

        # initialize factor matrices
        factors = []
        if args.load_tensor != '':
            for i in range(tensor.ndim):
                factors.append(tenpy.load_tensor_from_file(
                    args.load_tensor + 'mat' + str(i) + '.npy'
                ))
        elif args.hosvd != 0:
            if args.decomposition == "CP":
                for i in range(tensor.ndim):
                    factors.append(tenpy.random((args.hosvd_core_dim[i], rank_app)))
            elif args.decomposition == "Tucker":
                from tensor_decomposition.Tucker.common_kernels import hosvd
                factors = hosvd(tenpy, tensor, args.hosvd_core_dim, compute_core=False)
        else:
            if args.decomposition == "CP":
                for i in range(tensor.ndim):
                    factors.append(tenpy.random((tensor.shape[i], rank_app)))
            else:
                for i in range(tensor.ndim):
                    factors.append(tenpy.random((tensor.shape[i], args.hosvd_core_dim[i])))

        # make copies for each method
        factors_amdm_reduced = factors[:]
        factors_amdm_full = factors[:]
        factors_als = factors[:]
        factors_hybrid = factors[:]

        # run mahalanobis with reduced singular values
        csv_path = join(
            RESULTS_DIR,
            f'mahalanobis_reduced_{args.tensor}_order{args.order}_s{args.s}'
            f'_R{args.R}_Rapp{args.R_app}_iter{tensor_idx}.csv'
        )
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        args.reduce_val = 0
        if tenpy.is_master_proc():
            print(f"[info] running mahalanobis (reduced vals={args.num_vals})")
            for arg in vars(args):
                print(f"  {arg}: {getattr(args, arg)}")
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness', 'cond_num'])

        # note: cp_mahalanobis signature differs from original - would need adjustment
        # cp_mahalanobis(tenpy, factors_amdm_reduced, tensor, sparsity_pattern, num_iter, ...)

        # run full mahalanobis
        csv_path = join(
            RESULTS_DIR,
            f'mahalanobis_{args.tensor}_s{args.s}_R{args.R}_Rapp{args.R_app}_iter{tensor_idx}.csv'
        )
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        args.reduce_val = 0
        args.num_vals = rank_app
        if tenpy.is_master_proc():
            print(f"[info] running full mahalanobis")
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness', 'cond_num'])

        # run als
        csv_path = join(
            RESULTS_DIR,
            f'als_{args.tensor}_order{args.order}_s{args.s}'
            f'_R{args.R}_Rapp{args.R_app}_iter{tensor_idx}.csv'
        )
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        regularization = 1e-07
        args.method = 'DT'

        if tenpy.is_master_proc():
            print(f"[info] running als")
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness', 'cond_num'])

        # note: cp_als signature differs from original - would need adjustment
        # cp_als(tenpy, factors_als, tensor, sparsity_pattern, num_iter, ...)

        # run hybrid
        csv_path = join(
            RESULTS_DIR,
            f'hybrid_{args.tensor}_s{args.s}_R{args.R}_Rapp{args.R_app}_iter{tensor_idx}.csv'
        )
        is_new_log = not Path(csv_path).exists()
        csv_file = open(csv_path, 'a')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        args.reduce_val = 1
        args.num_vals = rank_app

        if tenpy.is_master_proc():
            print(f"[info] running hybrid")
            if is_new_log:
                csv_writer.writerow(['iterations', 'time', 'residual', 'fitness', 'cond_num'])

        # note: cp_mahalanobis signature differs from original - would need adjustment
        # cp_mahalanobis(tenpy, factors_hybrid, tensor, sparsity_pattern, num_iter, ...)

    print(f"\n[done] completed experiments for {num_tensors} tensors")
