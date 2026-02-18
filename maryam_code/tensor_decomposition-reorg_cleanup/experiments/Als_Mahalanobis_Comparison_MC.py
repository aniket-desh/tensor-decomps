import numpy as np
import sys
import time
import os
import csv
from pathlib import Path
from os.path import dirname, join
# correct_path = "/home/maryam/Documents/tensor_decomposition-reorg_cleanup_new4/tensor_decomposition-reorg_cleanup"
# if correct_path not in sys.path:
#     sys.path.insert(0, correct_path)
parent_dir = dirname(__file__)
tensor_dir = join(parent_dir, 'tensor_decomposition')

sys.path.insert(0, tensor_dir)
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors

import tensor_decomposition
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import argparse
import tensor_decomposition.utils.arg_defs as arg_defs
import csv
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.common_kernels import get_residual,get_residual_sp,compute_condition_number
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer, CP_PPALS_Optimizer
##########################################
import Generate_plots
#import error_computation
from scipy.linalg import svd
import numpy.linalg as la
import matplotlib.pyplot as plt
import copy
import random
from run_als_orgi import CP_ALS
from run_mahalanobis_orgi import CP_Mahalanobis
from generate_initial_guess import generate_initial_guess
from generate_input_tensor import generate_tensor
##########################################
parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')



############################################################################
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_amdm_arguments(parser)
    args, _ = parser.parse_known_args()

    
    s = args.s
    order = args.order
    R = args.R
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tensor = args.tensor
    tlib = args.tlib
    thresh = args.thresh
    ######################################
    epsilon = args.epsilon
    k = args.k 
    alpha = args.alpha
    num_runs = args.num_runs
    tol = args.tol
    type_noisy_tensor = args.type_noisy_tensor
    ######################################
    if args.R_app is None:
        R_app = args.R
    else:
        R_app = args.R_app
    if args.num_vals is None:
        args.num_vals = args.R
    

    if tlib == "numpy":
        import tensor_decomposition.backend.numpy_ext as tenpy
    elif tlib == "ctf":
        import tensor_decomposition.backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin();

    tenpy.seed(args.seed)
    RR = [10, 20, 30, 40, 50]
    best_run_residual_als = []
    best_run_residual_amdm = []
    best_run_residual_hybrid = []
    ####
    mean_residuals_als = []
    mean_residuals_amdm= []
    mean_residuals_hybrid = []
    ###
    std_residuals_als = []
    std_residuals_amdm = []
    std_residuals_hybrid = []
    for R in RR:
        args.R = R
        args.R_app = R
        args.r = R
        r = R
        all_residuals_als = []
        all_residuals_amdm = []
        all_residuals_hybrid = []
        final_residuals_als = []
        final_residuals_amdm = []
        final_residuals_hybrid = []
        
            #Generate a random tensor and initial factor matrices 
            #print(args)
        T_true,T, O, cov_empirical,cov_pinv_empirical, M_empirical_pinv= generate_tensor(tenpy, args)
            
            #Generate the initial factor matrices  
        for run in range(args.num_runs):
            A_ini = generate_initial_guess(tenpy, T, args)
            # print(A_ini[0].shape)
            # print(A_ini[1].shape)
            # print(A_ini[2].shape)
            
            # ALS Optimization
            
            # best_run_residual_als, best_run_norm_mahalanobis_als, final_residuals_als, mean_residuals_als, std_residuals_als = CP_ALS(tenpy, T_true,A_ini, T, O,args,cov_empirical,cov_pinv_empirical, M_empirical_pinv, csv_file=None, method='DT', res_calc_freq=1,
            # tol=1e-05)
            # A, residuals_als = CP_ALS(tenpy, T_true,A_ini, T, O,args,cov_empirical,cov_pinv_empirical, M_empirical_pinv, csv_file=None, method='DT', res_calc_freq=1,
            # tol=1e-05)
            
            A, residuals_als = CP_ALS(tenpy, A_ini, T, O,num_iter, args,R,R_app,r, csv_file=None,Regu=None, method='DT',res_calc_freq=1,
            tol=1e-05)
            # Mahalanobis Optimization
        
            # best_run_residual_amdm, best_run_norm_mahalanobis_amdm, final_residuals_amdm, mean_residuals_amdm, std_residuals_amdm =CP_Mahalanobis(tenpy,T_true, A_ini, T, O, args, cov_empirical,cov_pinv_empirical, M_empirical_pinv, thresh=None, csv_file=None, Regu=None, res_calc_freq=1)
            # B, residuals_amdm =CP_Mahalanobis(tenpy,T_true, A_ini, T, O, args, cov_empirical,cov_pinv_empirical, M_empirical_pinv, thresh=None, csv_file=None, Regu=None, res_calc_freq=1)

            B, residuals_amdm =CP_Mahalanobis(tenpy, A_ini, T, O,num_iter, args,R,R_app,r, thresh=None, csv_file=None, Regu=None, res_calc_freq=1)
            # Hybrid Algorithm
        
            # best_run_residual_hybrid, best_run_norm_mahalanobis_hybrid, final_residuals_hybrid, mean_residuals_hybrid, std_residuals_hybrid =CP_Mahalanobis(tenpy,T_true, A_ini, T, O, args, cov_empirical,cov_pinv_empirical, M_empirical_pinv, thresh=10.0, csv_file=None, Regu=None, res_calc_freq=1)
            
            # C, residuals_hybrid =CP_Mahalanobis(tenpy,T_true, A_ini, T, O, args, cov_empirical,cov_pinv_empirical, M_empirical_pinv, thresh=10.0, csv_file=None, Regu=None, res_calc_freq=1)

            C, residuals_hybrid =CP_Mahalanobis(tenpy, A_ini, T, O,num_iter, args,R,R_app,r, thresh=10.0, csv_file=None, Regu=None, res_calc_freq=1)
            
            # Data storage
            all_residuals_als.append(residuals_als)
            all_residuals_amdm.append(residuals_amdm)
            all_residuals_hybrid.append(residuals_hybrid)
            ###
            final_residuals_als.append(residuals_als[-1])
            final_residuals_amdm.append(residuals_amdm[-1])
            final_residuals_hybrid.append(residuals_hybrid[-1])
            ###
        best_run_index_als = np.argmin(final_residuals_als)
        best_run_index_amdm = np.argmin(final_residuals_amdm)
        best_run_index_hybrid = np.argmin(final_residuals_hybrid)
        ###
        mean_residuals_als.append(np.mean(final_residuals_als, axis=0))
        mean_residuals_amdm.append(np.mean(final_residuals_amdm, axis=0))
        mean_residuals_hybrid.append(np.mean(final_residuals_hybrid, axis=0))
        ###
        std_residuals_als.append(np.std(final_residuals_als, axis=0))
        std_residuals_amdm.append(np.std(final_residuals_amdm, axis=0))
        std_residuals_hybrid.append(np.std(final_residuals_hybrid, axis=0))
        ###
        best_run_residual_als.append(all_residuals_als[best_run_index_als])
        best_run_residual_amdm.append(all_residuals_amdm[best_run_index_amdm])
        best_run_residual_hybrid.append(all_residuals_hybrid[best_run_index_hybrid])
        ###
    residuals = [best_run_residual_als , best_run_residual_amdm, best_run_residual_hybrid]
            # norm_mahalanobis = [best_run_norm_mahalanobis_als, best_run_norm_mahalanobis_amdm,best_run_norm_mahalanobis_hybrid]
    final_residuals = [final_residuals_als, final_residuals_amdm,final_residuals_hybrid]
    mean_residuals = [mean_residuals_als, mean_residuals_amdm,mean_residuals_hybrid]
    std_residuals = [std_residuals_als, std_residuals_amdm,std_residuals_hybrid]
    print(residuals)
    print(final_residuals)
    print(mean_residuals)
    print(std_residuals)
        
    #plot results
    Generate_plots.plot_results(residuals, final_residuals, mean_residuals, std_residuals, s, RR, epsilon, alpha)


    
