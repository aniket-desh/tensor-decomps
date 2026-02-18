"""
multiple tensor experiment comparing mahalanobis and als methods.
runs experiments across multiple randomly generated noisy tensors,
using shared initial factors for fair method comparison.
"""

import numpy as np
import sys
import time
import os
import argparse
import csv
import copy
from pathlib import Path
from os.path import dirname, join

import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.utils.arg_defs as arg_defs
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

    os.makedirs(RESULTS_DIR, exist_ok=True)

    num_tensors = 5
    dims = [size] * order

    for tensor_idx in range(num_tensors):
        print(f"\n[info] processing tensor {tensor_idx + 1}/{num_tensors}")

        seed = args.seed + tensor_idx
        sparsity_pattern = None

        # generate noisy tensor with covariance info
        if tensor_type == "noisy_tensor":
            if args.type_noisy_tensor == "old_model":
                tensor_true, tensor, factors_true, cov_empirical, cov_pinv_empirical, m_empirical_pinv = \
                    synthetic_tensors.generate_tensor_with_noise_old_model(
                        tenpy, dims, rank, args.k, args.epsilon, args.alpha, seed=seed
                    )
            else:
                tensor_true, tensor, factors_true, cov_empirical, cov_pinv_empirical, m_empirical_pinv = \
                    synthetic_tensors.generate_tensor_with_noise_new_model(
                        tenpy, dims, rank, args.k, args.epsilon, args.alpha, seed=seed
                    )
        elif tensor_type == "random_col":
            tensor_true, tensor, factors_true, cov_empirical, cov_pinv_empirical, m_empirical_pinv = \
                synthetic_tensors.collinearity_tensor(
                    tenpy, size, order, rank, args.k, args.epsilon, args.col, seed=seed
                )
        else:
            raise ValueError(
                f"tensor type '{tensor_type}' not supported for mahalanobis comparison. "
                f"use 'noisy_tensor' or 'random_col'."
            )

        tenpy.printf(f"[info] tensor shape: {tensor.shape}")

        # generate shared initial factors for fair comparison across methods
        np.random.seed(seed * 2001)
        initial_factors = [np.random.rand(tensor.shape[i], rank_app) for i in range(tensor.ndim)]

        # --- run mahalanobis with reduced singular values ---
        csv_path = join(
            RESULTS_DIR,
            f'mahalanobis_reduced_{tensor_type}_order{order}_s{size}'
            f'_R{rank}_Rapp{rank_app}_iter{tensor_idx}.csv'
        )
        csv_file = open(csv_path, 'w', newline='')

        args.reduce_val = 0
        if tenpy.is_master_proc():
            print(f"[info] running mahalanobis (reduced vals={args.num_vals})")
            for arg in vars(args):
                print(f"  {arg}: {getattr(args, arg)}")

        cp_mahalanobis(
            tenpy, tensor_true, initial_factors, tensor, sparsity_pattern, args,
            cov_empirical, cov_pinv_empirical, m_empirical_pinv,
            thresh=args.thresh, csv_file=csv_file,
            regularization=args.regularization, res_calc_freq=args.res_calc_freq
        )
        csv_file.close()

        # --- run full mahalanobis ---
        csv_path = join(
            RESULTS_DIR,
            f'mahalanobis_full_{tensor_type}_order{order}_s{size}'
            f'_R{rank}_Rapp{rank_app}_iter{tensor_idx}.csv'
        )
        csv_file = open(csv_path, 'w', newline='')

        args.reduce_val = 0
        args.num_vals = rank_app
        if tenpy.is_master_proc():
            print(f"[info] running full mahalanobis")

        cp_mahalanobis(
            tenpy, tensor_true, initial_factors, tensor, sparsity_pattern, args,
            cov_empirical, cov_pinv_empirical, m_empirical_pinv,
            thresh=args.thresh, csv_file=csv_file,
            regularization=args.regularization, res_calc_freq=args.res_calc_freq
        )
        csv_file.close()

        # --- run als ---
        csv_path = join(
            RESULTS_DIR,
            f'als_{tensor_type}_order{order}_s{size}'
            f'_R{rank}_Rapp{rank_app}_iter{tensor_idx}.csv'
        )
        csv_file = open(csv_path, 'w', newline='')

        args.method = 'DT'
        if tenpy.is_master_proc():
            print(f"[info] running als")

        cp_als(
            tenpy, tensor_true, initial_factors, tensor, sparsity_pattern, args,
            cov_empirical, cov_pinv_empirical, m_empirical_pinv,
            csv_file=csv_file, regularization=args.regularization,
            method='DT', res_calc_freq=args.res_calc_freq
        )
        csv_file.close()

        # --- run hybrid (mahalanobis with value reduction) ---
        csv_path = join(
            RESULTS_DIR,
            f'hybrid_{tensor_type}_order{order}_s{size}'
            f'_R{rank}_Rapp{rank_app}_iter{tensor_idx}.csv'
        )
        csv_file = open(csv_path, 'w', newline='')

        args.reduce_val = 1
        args.num_vals = rank_app
        if tenpy.is_master_proc():
            print(f"[info] running hybrid")

        cp_mahalanobis(
            tenpy, tensor_true, initial_factors, tensor, sparsity_pattern, args,
            cov_empirical, cov_pinv_empirical, m_empirical_pinv,
            thresh=args.thresh, csv_file=csv_file,
            regularization=args.regularization, res_calc_freq=args.res_calc_freq
        )
        csv_file.close()

    print(f"\n[done] completed experiments for {num_tensors} tensors")
