import numpy as np
import sympy as sp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import argparse
from scipy.linalg import svd
from basis_functions import generate_b_spline_basis, chebyshev_polynomial, fourier_basis, legendre_polynomial, legendre_polynomial_scaled, chebyshev_polynomial_scaled, gaussian_basis
from integration import compute_M1_M2, compute_Mi
from sampling import rejection_sampling_positions, sample_velocities
from analythical_functions import rho_1d, u_1d, T_1d, rho_2d, u_12d, u_22d, T_2d, rho_3d, u_13d, u_23d, u_33d, T_3d, analytical_maxwellian, maxwellian_ana, analytical_rho
from reconstruction_c_hat_MW import compute_C_hat, reconstructed_maxwellian, reconstructed_maxwellian_ana, svd_decomposition_truncated, reconstructed_particle_density
from arg_def import parse_arguments
import tensor_decomposition.backend.numpy_ext as tenpy
from scipy.integrate import simpson
################################################################################
import sys
import time
import os
import csv
from pathlib import Path
from os.path import dirname, join
parent_dir = dirname(__file__)
tensor_dir = join(parent_dir, 'tensor_decomposition')

sys.path.insert(0, tensor_dir)
#import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors

import tensor_decomposition
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import tensor_decomposition.utils.arg_defs as arg_defs
import csv
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.common_kernels import get_residual,get_residual_sp,compute_condition_number
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer, CP_PPALS_Optimizer
import Generate_plots
#import error_computation
from scipy.linalg import svd
import numpy.linalg as la
import matplotlib.pyplot as plt
import copy
import random
from run_als_org import CP_ALS
from run_mahalanobis_org import CP_Mahalanobis
from generate_initial_guess import generate_initial_guess
from generate_input_tensor import generate_tensor
parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')
##########################################


def main():



    # symbolic variables
    x = sp.symbols('x')
    y = sp.symbols('y')
    z = sp.symbols('z')
    symbol_spatial = [x, y, z]
    vx = sp.symbols('vx')
    vy = sp.symbols('vy')
    vz = sp.symbols('vz')
    symbol_v = [vx, vy, vz]
    
    parser = argparse.ArgumentParser()
    #arg_def.parse_arguments(parser)
    args = parse_arguments()
#####################################################################################
    arg_defs.add_general_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_amdm_arguments(parser)


    #     parser = argparse.ArgumentParser()
    # arg_defs.add_general_arguments(parser)
    # arg_defs.add_sparse_arguments(parser)
    # arg_defs.add_pp_arguments(parser)
    # arg_defs.add_col_arguments(parser)
    # args, _ = parser.parse_known_args()
        #Set up CSV logging
    csv_path = join(results_dir, 'Mahalanobis-'+str(args.order)+'-s-'+str(args.s)+'-R-'
        +str(args.R)+'-R_app-'+str(args.R_app)+'-thresh-'+str(args.thresh)+'.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')#, newline='')
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    s = args.s
    order = args.order
    R = args.R
    num_iter = args.num_iter
    sp_frac = args.sp_fraction
    tlib = args.tlib
    thresh = args.thresh
    if args.R_app is None:
        R_app = args.R
    else:
        R_app = args.R_app
    if args.num_vals is None:
        args.num_vals = args.R
    

    # if tlib == "numpy":
    #     import tensor_decomposition.backend.numpy_ext as tenpy
    # elif tlib == "ctf":
    #     import tensor_decomposition.backend.ctf_ext as tenpy
    #     import ctf
    #     tepoch = ctf.timer_epoch("ALS")
    #     tepoch.begin();
    args.tlib = "numpy"  

    # if tenpy.is_master_proc():
    #     # print the arguments
    #     for arg in vars(args) :
    #         print( arg+':', getattr(args, arg))
    #     # initialize the csv file
    #     if is_new_log:
    #         csv_writer.writerow([
    #             'method','iterations', 'time', 'residual', 'fitness','cond_num'
    #         ])

    #tenpy.seed(args.seed)
    ################################################################################
    
    # Spatial domain parameters
    x_a = args.x_a
    x_b = args.x_b
    y_a = args.y_a
    y_b = args.y_b
    z_a = args.z_a
    z_b = args.z_b
    num_internal_knots_x = args.num_internal_knots_x
    num_internal_knots_y = args.num_internal_knots_y
    num_internal_knots_z = args.num_internal_knots_z
    degree_x = args.degree_x
    degree_y = args.degree_y
    degree_z = args.degree_z
    
    interval_x = [(x_a,x_b), (y_a,y_b), (z_a, z_b)]
    num_internal_knots_spatial = [num_internal_knots_x, num_internal_knots_y, num_internal_knots_z]
    degree_spatial = [degree_x, degree_y, degree_z]
    
    # Velocity domain parameters
    vx_a = args.vx_a
    vx_b = args.vx_b
    vy_a = args.vy_a
    vy_b = args.vy_b
    vz_a = args.vz_a
    vz_b = args.vz_b
    num_internal_knots_vx = args.num_internal_knots_vx
    num_internal_knots_vy = args.num_internal_knots_vy
    num_internal_knots_vz = args.num_internal_knots_vz
    degree_vx = args.degree_vx
    degree_vy = args.degree_vy
    degree_vz = args.degree_vz
    
    interval_v = [(vx_a,vx_b), (vy_a,vy_b), (vz_a, vz_b)]
    num_internal_knots_v = [num_internal_knots_vx, num_internal_knots_vy, num_internal_knots_vz]
    degree_v = [degree_vx, degree_vy, degree_vz]
    # Sampling parameters
    number_samples = args.number_samples

    # Basis functions
    basis_functions_spatial = args.basis_functions_spatial
    basis_functions_velocity = args.basis_functions_velocity
    
    #Dimentionality of spatial and velocity
    D = args.D
    
    # Grid parameters
    x_grid_size = args.x_grid_size
    v_grid_size = args.v_grid_size

    # Generate spatial and velocity grids
    x_grid = np.linspace(x_a, x_b, x_grid_size)
    y_grid = np.linspace(y_a, y_b, x_grid_size)
    z_grid = np.linspace(z_a, z_b, x_grid_size)
    vx_grid = np.linspace(vx_a, vx_b, v_grid_size)
    vy_grid = np.linspace(vy_a, vy_b, v_grid_size)
    vz_grid = np.linspace(vz_a, vz_b, v_grid_size)

    # Basis function #####################################################################
    # Spatial
    phi_basis_x = []
    knots_x = []
    M1 = []
    if args.basis_functions_spatial == "B-spline":
        for i in range(D):
            phi_basis, knots = generate_b_spline_basis(interval_x[i][0], interval_x[i][1], num_internal_knots_spatial[i], degree_spatial[i], symbol_spatial[i]) 
            phi_basis_x.append(phi_basis)
            knots_x.append(knots)
            M1_i = compute_M1_M2(phi_basis, knots, degree_spatial[i], symbol_spatial[i]) # Compute M1 (spaatial matrix) symbolically
            M1.append(M1_i)
    elif args.basis_functions_spatial == "Chebyshev":
        for i in range(D):
            phi_basis = [chebyshev_polynomial(symbol_spatial[i], j) for j in range(degree_spatial[i])]
            phi_basis_x.append(phi_basis)
            M1_i = compute_Mi(phi_basis, degree_spatial[i], symbol_spatial[i], interval_x[i][0], interval_x[i][1])
            M1.append(M1_i)
    elif args.basis_functions_spatial == "Legendre":
        for i in range(D):
            phi_basis = [legendre_polynomial(symbol_spatial[i], j) for j in range(degree_spatial[i])]
            #[legendre_polynomial_scaled(symbol_spatial[i], j, interval_x[i][0], interval_x[i][1]) for j in range(degree_v[i])]
    
            phi_basis_x.append(phi_basis)
            M1_i = compute_Mi(phi_basis, degree_spatial[i], symbol_spatial[i], interval_x[i][0], interval_x[i][1])
            M1.append(M1_i)    
    elif args.basis_functions_spatial == "Fourier":
        for i in range(D):
            phi_basis = [fourier_basis(symbol_spatial[i], j) for j in range(degree_spatial[i])]
            phi_basis_x.append(phi_basis)
            M1_i = compute_Mi(phi_basis, degree_spatial[i], symbol_spatial[i], interval_x[i][0], interval_x[i][1])
            M1.append(M1_i)
    elif args.basis_functions_spatial == "Gaussian":
        for i in range(D):
            phi_basis = gaussian_basis(symbol_spatial[i], degree_spatial[i],interval_x[i][0], interval_x[i][1]) 
            phi_basis_x.append(phi_basis)
            M1_i = compute_Mi(phi_basis, degree_spatial[i], symbol_spatial[i], interval_x[i][0], interval_x[i][1])
            M1.append(M1_i)
    # Velocity
    psi_basis_v = []
    knots_v = []
    M2 = []
    if args.basis_functions_velocity == "B-spline":
        for i in range(D):
            psi_basis, knots = generate_b_spline_basis(interval_v[i][0], interval_v[i][1], num_internal_knots_v[i], degree_v[i], symbol_v[i]) 
            psi_basis_v.append(psi_basis)
            knots_v.append(knots)
            M2_i = compute_M1_M2(psi_basis, knots, degree_v[i], symbol_v[i]) # Compute M2 (velocity matrix) symbolically
            M2.append(M2_i)   
    elif args.basis_functions_velocity == "Chebyshev":
        for i in range(D):
            psi_basis = [chebyshev_polynomial_scaled(symbol_v[i], j, interval_v[i][0], interval_v[i][1]) for j in range(degree_v[i])]
            #psi_basis = [chebyshev_polynomial(symbol_v[i], j) for j in range(degree_v[i])]
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)
    elif args.basis_functions_velocity == "Legendre":
        for i in range(D):
            psi_basis = [legendre_polynomial_scaled(symbol_v[i], j, interval_v[i][0], interval_v[i][1]) for j in range(degree_v[i])]
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)
    elif args.basis_functions_velocity == "Fourier":    
        for i in range(D):
            psi_basis = [fourier_basis(symbol_v[i], j) for j in range(degree_v[i])]
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)
    elif args.basis_functions_velocity == "Gaussian":
        for i in range(D):
            psi_basis = gaussian_basis(symbol_v[i], degree_v[i],interval_v[i][0], interval_v[i][1]) 
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)        
    # print(M1)
    # print(M2)
################################################################################
    sample_sizes = [10, 10000, 1000000]
    #sample_sizes = [1000000]
    err_L1 = []
    err_L2 = []
    err_L1_rho = []
    err_L2_rho = []
    err_rho = []
    err = []

    phi_funcs = []
    psi_funcs = []

    for d in range(D):
        phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis_x[d]]
        psi_fun = [sp.lambdify(symbol_v[d], psi, 'numpy') for psi in psi_basis_v[d]]
        phi_funcs.append(phi_fun)
        psi_funcs.append(psi_fun)
        
    for number_samples in sample_sizes:
        # Sample data 
        _, x_samples = rejection_sampling_positions(interval_x, number_samples, D)
        v_samples = sample_velocities(x_samples, D)
        # x_samples = np.array(x_samples)
        # v_samples = np.array(v_samples)
        # print(x_samples)
        # print(v_samples)
        # print(phi_basis_x)
        # print(psi_basis_v)
        # Compute C_hat 
        C_hat = compute_C_hat(phi_basis_x, psi_basis_v, x_samples, v_samples, D)
        #print((C_hat))
        U, sigma, VT = np.linalg.svd(C_hat, full_matrices=False)
        # print(sigma)
        # print((M1))
        # print((M2))
        # Compute C
        if D == 1:
            #C = np.linalg.inv(M1).T @ C_hat @ np.linalg.inv(M2)
            C = np.einsum('ki,kl,lj->ij', np.linalg.inv(M1[0]), C_hat, np.linalg.inv(M2[0]))
        elif D == 2:
            C = np.einsum('ia,jb,ijkl,kc,ld->abcd', np.linalg.inv(M1[0]), np.linalg.inv(M1[1]), C_hat, np.linalg.inv(M2[0]), np.linalg.inv(M2[1]))
        elif D == 3:
            C = np.einsum('ia,jb,kc,ijklmn,ld,me,nf->abcdef', np.linalg.inv(M1[0]), np.linalg.inv(M1[1]), np.linalg.inv(M1[2]), C_hat, np.linalg.inv(M2[0]), np.linalg.inv(M2[1]),  np.linalg.inv(M2[2])) 
            
        
    ###########################################################
        # # Compute error 
        # d_v = 1
        # rho_a = 1.0
        # delta = 5e-3  
        # u_a = delta * sp.sin(2 * sp.pi * x) 
        # #u_a = delta * x
        # T_a = 1.0 + 0.001 * x  
        #  ana_MW = maxwellian_ana(d_v, rho_a, u_a, T_a, v)
        #  rec_MW = reconstructed_maxwellian_ana(phi_basis_x, psi_basis_v, C)
        # # #print(f"ana_MW: {ana_MW}")
        # # #print(f"rec_MW: {rec_MW}")
        #  error = sp.Abs(ana_MW - rec_MW)
        # # #print(error)
        #  numerical_integrand = sp.lambdify((x, v), error, modules='numpy')
    
        # #  SciPy's dblquad
        # result, _ = dblquad(
        #     lambda v, x: numerical_integrand(x, v),  
        #     x_a, x_b,                   # x_bound
        #     lambda _: vx_a,           # v lower_bound
        #     lambda _: vx_b,            # v upper_bound
        #       epsabs=1.49e-08,
        #       epsrel=1.49e-08
        # )
        # print(f"L1 Error: {result:.8e}")
    ##################################################################
                
        x_grid = np.linspace(x_a, x_b, x_grid_size)
        y_grid = np.linspace(y_a, y_b, x_grid_size)
        z_grid = np.linspace(z_a, z_b, x_grid_size)
        vx_grid = np.linspace(vx_a, vx_b, v_grid_size)
        vy_grid = np.linspace(vy_a, vy_b, v_grid_size)
        vz_grid = np.linspace(vz_a, vz_b, v_grid_size)
        y_fix = 0.0
        vy_fix = 0.0
        z_fix = 0.0
        vz_fix = 0.0
        XX, VV = np.meshgrid(x_grid, vx_grid, indexing='ij')
        if D == 1:
            X, V = np.meshgrid(x_grid, vx_grid, indexing='ij')
            analytical_values_1d = [
                analytical_maxwellian(rho_1d(x_i), D, u=[u_1d(x_i)], T=T_1d(x_i), ve=velocity)
                for x_i, velocity in zip(x_grid, vx_grid)]
           # print(analytical_values_1d)
            reconstructed_values_1d =  [reconstructed_maxwellian(phi_funcs, psi_funcs, x_i, v_j, C, D) for x_i, v_j in zip(x_grid, vx_grid)]
            #reconstructed_values_1d = np.array([reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C, D) for x_i, v_j in zip(x_grid, vx_grid)]) 
            
            analytical_values_2d = np.array([
                    analytical_maxwellian(rho_1d(x_i), D, u=u_1d(x_i), T=T_1d(x_i), ve=v_j)
                    for x_i, v_j in zip(X.flatten(), V.flatten())
                ]).reshape(X.shape)
            #print("Sizes are", analytical_values_2d.shape, X.shape, V.shape, x_grid.shape, vx_grid.shape)
            #print(reconstructed_values_1d)
            reconstructed_values_2d = np.array([
            reconstructed_maxwellian(phi_funcs, psi_funcs, x_i, v_j, C, D)
            for x_i, v_j in zip(X.flatten(), V.flatten())]).reshape(X.shape)

            # analytical_array = np.array(analytical_values_1d)
            # reconstructed_array = np.array(reconstructed_values_1d)
            analytical_array = analytical_values_2d
            reconstructed_array = reconstructed_values_2d
            error = (np.linalg.norm(analytical_array.astype(np.float64) - reconstructed_array.astype(np.float64)))/np.sqrt(len(analytical_array.flatten()))
            err.append(error)
            
            # error = analytical_array - reconstructed_array
            # n = len(analytical_array)
            # dx = x_grid[1] - x_grid[0] 
            # dv = vx_grid[1] - vx_grid[0]
            # weights_x = np.ones(len(x_grid))
            # weights_x[0] = weights_x[-1] = 0.5  # half weights at endpoints
            # weights_v = np.ones(len(vx_grid))
            # weights_v[0] = weights_v[-1] = 0.5  # 1D trapezoidal weights for v
            # weights_2d = np.outer(weights_x, weights_v)  
            # error_L2 = np.sqrt(np.sum(weights_2d * error**2) * dx * dv)
            # # L1 norm (absolute error with quadrature)
            # error_L1 = np.sum(weights_2d * np.abs(error)) * dx * dv
            # print(f"L1 error: {error_L1}")
            # print(f"L2 error: {error_L2}")
            # err_L1.append(error_L1)
            # err_L2.append(error_L2)
            print(f" error: {error}")
            ####################################################
            # Integrate reconstructed Maxwellian over velocity to obtain particle density
            # rho_reconstructed = np.sum(reconstructed_values_2d, axis=0) * dv  
            # rho_analytical = np.array([rho_1d(x_i) for x_i in x_grid])  
            rho_reconstructed = reconstructed_particle_density(phi_basis_x, psi_basis_v, C, symbol_v[0], symbol_v[1], interval_v[0][0], interval_v[0][1], D)
            compute_rec = sp.lambdify(x, rho_reconstructed, 'numpy')
            rho_reconstructed_vals = compute_rec(x_grid)  
            #print(rho_reconstructed_vals)
            rho_analytical_vals = np.array([analytical_rho(rho_1d(x_i), D, u_1d(x_i), T_1d(x_i), symbol_v[0], interval_v[0][0], interval_v[0][1]) for x_i in x_grid])
            # compute_ana = sp.lambdify(x, rho_analytical, 'numpy')
            # rho_analytical_vals = compute_ana(x_grid)  
            #print(rho_analytical_vals)
            error_rho = np.linalg.norm(rho_analytical_vals.astype(np.float64) - rho_reconstructed_vals.astype(np.float64))/np.sqrt(len(rho_analytical_vals.flatten()))
            
            # # L1 error (integrate absolute error over space)
            # error_L1_rho = np.sum(0.5 * (error_rho[:-1] + error_rho[1:]) * dx)
            
            # # L2 error (integrate squared error over space)
            # error_L2_rho = np.sqrt(np.sum(0.5 * (error_rho[:-1]**2 + error_rho[1:]**2) * dx))
            # err_L1_rho.append(error_L1_rho)
            # err_L2_rho.append(error_L2_rho)
            err_rho.append(error_rho)
            print(f"error reconstructed rho: {error_rho}")
            #print(f"L2 error reconstructed rho: {error_L2_rho}")
########################################################################
            # U, sigma, VT = np.linalg.svd(C, full_matrices=False)
            # #print(C)
            # print(sigma)
            # max_rank = min(C.shape)
            # ranks = list(range(1, max_rank + 1))
            # errors1_L1 = []
            # errors2_L1 = []
            # for r in ranks:
            #     C_r = svd_decomposition_truncated(U, sigma, VT, r)
            #     if r == 3:
            #         C_app = C_r 
            #     reconstructed_values_1d_C_r =  np.array([reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C_r, D) for x_i, v_j in zip(X.flatten(), V.flatten())]).reshape(X.shape)
            #     reconstructed_array_C_r = reconstructed_values_1d_C_r
            #     error1 = analytical_array - reconstructed_array_C_r
            #     error2 = reconstructed_array - reconstructed_array_C_r

            #     errors1_L1.append(dx * dv* np.sum(weights_2d * np.abs(error1)))
            #     errors2_L1.append(dx * dv* np.sum(weights_2d * np.abs(error2)))

            # plt.figure(figsize=(10, 5))
            # plt.subplot()
            # plt.plot(ranks, errors1_L1, label='L1 Error (M - f_c_bar)')
            # plt.plot(ranks, errors2_L1, label='L1 Error (f - f_c_bar)')
            # plt.xlabel('Rank')
            # plt.ylabel('Error')
            # plt.title('Error Convergence')
            # plt.legend()
            # plt.xscale('log')
            # reconstructed_values_2d_c_r = np.array([
            # reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C_app, D)
            # for x_i, v_j in zip(X.flatten(), V.flatten())]).reshape(X.shape)
###############################################################################                
            
            
        
        elif D == 2:
            analytical_values_1d = [
                analytical_maxwellian(rho_2d(x_i, y_i), D, u=[u_12d(y_i), u_22d(x_i)], T=T_2d(x_i, y_i), ve=np.array([v_x, v_y]))
                for x_i, y_i, v_x, v_y in zip(x_grid, y_grid, vx_grid, vy_grid)]
            
            reconstructed_values_1d =  [reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_i]), np.array([v_x, v_y]), C, D) for x_i, y_i, v_x, v_y in zip(x_grid, y_grid, vx_grid, vy_grid)]

            analytical_values_2d = np.array([
            analytical_maxwellian(rho_2d(x_i, y_fix), D, u=[u_12d(y_fix), u_22d(x_i)], T=T_2d(x_i, y_fix), ve=np.array([v_j, vy_fix]))
            for x_i, v_j in zip(XX.flatten(), VV.flatten())
        ]).reshape(XX.shape)
            #print(analytical_values_2d.shape)    
            reconstructed_values_2d = np.array([
            reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_fix]), np.array([v_j, vy_fix]), C, D)
            for x_i, v_j in zip(XX.flatten(), VV.flatten())
        ]).reshape(XX.shape)
            X, Y, VX, VY = np.meshgrid(x_grid, y_grid, vx_grid, vy_grid, indexing='ij')

            analytical_values_3d = np.array([
            analytical_maxwellian(rho_2d(x_i, y_i), D, u=[u_12d(y_i), u_22d(x_i)], T=T_2d(x_i, y_i), ve=np.array([vx_j, vy_j]))
            for x_i, y_i, vx_j, vy_j in zip(X.flatten(), Y.flatten(), VX.flatten(), VY.flatten())
        ]).reshape(X.shape)
        
            reconstructed_values_3d = np.array([
            reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_i]), np.array([vx_j, vy_j]), C, D)
            for x_i, y_i, vx_j, vy_j in zip(X.flatten(), Y.flatten(), VX.flatten(), VY.flatten())
        ]).reshape(X.shape)
            
            # analytical_array = np.array(analytical_values_1d)
            # reconstructed_array = np.array(reconstructed_values_1d)
            analytical_array = np.array(analytical_values_3d)
            reconstructed_array = np.array(reconstructed_values_3d)
            #error = np.linalg.norm(analytical_array - reconstructed_array)
            error = (np.linalg.norm(analytical_array.astype(np.float64) - reconstructed_array.astype(np.float64)))/np.sqrt(len(analytical_array.flatten()))
            
            err.append(error)
            print(f"error: {error}")
            # error = analytical_array - reconstructed_array
            # dx = x_grid[1] - x_grid[0]  
            # dy = y_grid[1] - y_grid[0]  
            # dvx = vx_grid[1] - vx_grid[0]  
            # dvy = vy_grid[1] - vy_grid[0]  
            
            # # Trapezoidal weights for each dimension
            # weights_x = np.ones(len(x_grid))
            # weights_x[0] = weights_x[-1] = 0.5  
            # weights_y = np.ones(len(y_grid))
            # weights_y[0] = weights_y[-1] = 0.5  
            # weights_vx = np.ones(len(vx_grid))
            # weights_vx[0] = weights_vx[-1] = 0.5  
            # weights_vy = np.ones(len(vy_grid))
            # weights_vy[0] = weights_vy[-1] = 0.5  
            
            # weights_4d = np.einsum('i,j,k,l->ijkl', weights_x, weights_y, weights_vx, weights_vy)
            
            # # L1 error (integrate absolute error )
            # error_L1 = np.sum(weights_4d * np.abs(error)) * dx * dy * dvx * dvy
            
            # # L2 error (integrate squared error over 4D grid)
            # error_L2 = np.sqrt(np.sum(weights_4d * error**2) * dx * dy * dvx * dvy)
            # print(f"L1 error: {error_L1}")
            # print(f"L2 error: {error_L2}")
            # err_L1.append(error_L1)
            # err_L2.append(error_L2)
##############################################################
            #  # Integrate reconstructed Maxwellian over velocity to obtain particle density
            # # rho_reconstructed = np.sum(reconstructed_values_3d, axis=(2, 3)) * dvx * dvy  
            # # rho_analytical = np.array([[rho_2d(x, y) for y in y_grid] for x in x_grid]) 
            # rho_reconstructed = reconstructed_particle_density(phi_basis_x, psi_basis_v, C, symbol_v[0], symbol_v[1], interval_v[0][0], interval_v[0][1], D)
            # compute = sp.lambdify((x, y), rho_reconstructed, 'numpy')
            # X_1d, Y_1d = np.meshgrid(x_grid, y_grid)
            # rho_reconstructed_vals = compute(X_1d, Y_1d)
            # #rho_reconstructed_vals = compute(x_grid, y_grid)  # Vectorized evaluation

            # #print(rho_reconstructed_vals)
            # rho_analytical_vals = np.array([analytical_rho(rho_2d(x_i, y_i), D, [u_12d(y_i), u_22d(x_i)], T_2d(x_i, y_i), [symbol_v[0], symbol_v[1]], interval_v[0][0], interval_v[0][1]) for x_i, y_i in zip(X_1d.flatten(), Y_1d.flatten())]).reshape(X_1d.shape)
            # #print(rho_analytical_vals)
            # #error_rho = np.abs(rho_analytical_vals - rho_reconstructed_vals)
            # # x_weights = np.ones(len(x_grid))
            # # x_weights[0] = x_weights[-1] = 0.5  
            # # y_weights = np.ones(len(y_grid))
            # # y_weights[0] = y_weights[-1] = 0.5  
            # # weights_2d = np.outer(x_weights, y_weights) 
            
            # # # L1 error (integrate absolute error)
            # # error_L1_rho = np.sum(weights_2d * error_rho) * dx * dy
            
            # # # L2 error (integrate squared error)
            # # error_L2_rho = np.sqrt(np.sum(weights_2d * error_rho**2) * dx * dy)


            # error_rho = (np.linalg.norm(rho_analytical_vals.astype(np.float64) - rho_reconstructed_vals.astype(np.float64)))/np.sqrt(len(rho_analytical_vals.flatten())) 
            
            # # err_L1_rho.append(error_L1_rho)
            # # err_L2_rho.append(error_L2_rho)
            # err_rho.append(error_rho)
            # # print(f"L1 error reconstructed rho: {error_L1_rho}")
            # print(f"error rho: {error_rho}")
#########################################################################
            I, J, K, L = C.shape 
            C_unfold = C.reshape(I* J, K, L)
            A_ini = generate_initial_guess(tenpy, C_unfold, args)
            #C_folded = C_unfold.reshape(I, J, K, L)
            #ranks = [5, 10, 20, 30, 50, 100]
            ranks = [10]
            errors1_L1 = []
            errors2_L1 = []
            O = None 
            for R in ranks:
                #print(args)
                
                B, res_als = CP_ALS(tenpy, A_ini, C_unfold, O, num_iter, csv_file=None, Regu=None, method='DT', args=args, res_calc_freq=1, tol=1e-05)
                
                A_ini = generate_initial_guess(tenpy, C_unfold, args)
                A, res_amdm = CP_Mahalanobis(tenpy, A_ini, C_unfold, O, num_iter, csv_file=None, Regu=None, args=args, res_calc_freq=1)
                C_r_unfold = tenpy.einsum('ir, jr, kr-> ijk', *A)
                C_r = C_r_unfold.reshape(I, J, K, L)
                C_r_unfold_als = tenpy.einsum('ir, jr, kr-> ijk', *B)
                C_r_als = C_r_unfold_als.reshape(I, J, K, L)
                if R == 10:
                    C_app = C_r 
                    C_app_als = C_r_als
                reconstructed_values_1d_C_r =  np.array([reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_i]), np.array([v_x, v_y]), C_r, D) for x_i, y_i, v_x, v_y in zip(X.flatten(), Y.flatten(), VX.flatten(), VY.flatten())
        ]).reshape(X.shape)

                reconstructed_values_1d_C_r_als =  np.array([reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_i]), np.array([v_x, v_y]), C_r_als, D) for x_i, y_i, v_x, v_y in zip(X.flatten(), Y.flatten(), VX.flatten(), VY.flatten())
        ]).reshape(X.shape)
                reconstructed_array_C_r = np.array(reconstructed_values_1d_C_r)
                reconstructed_array_C_r_als = np.array(reconstructed_values_1d_C_r_als)
                
                error1 = analytical_array - reconstructed_array_C_r
                error2 = reconstructed_array - reconstructed_array_C_r
                errors1_L2 = (np.linalg.norm(analytical_array.astype(np.float64) - reconstructed_array_C_r.astype(np.float64)))/np.sqrt(len(analytical_array.flatten()))
                errors2_L2 = (np.linalg.norm(reconstructed_array .astype(np.float64) - reconstructed_array_C_r.astype(np.float64)))/np.sqrt(len(analytical_array.flatten()))
                print(f"errors1_L2: {errors1_L2}")
                print(f"errors2_L2: {errors2_L2}")
                # errors1_L1.append(dx * dy* dvx* dvy* np.sum(weights_4d * np.abs(error1)))
                # errors2_L1.append(dx * dy* dvx* dvy* np.sum(weights_4d * np.abs(error2)))
            # plt.figure(figsize=(10, 5))
            # plt.subplot()
            # plt.plot(ranks, errors1_L1, label='L2 Error (M - f_c_bar)')
            # plt.plot(ranks, errors2_L1, label='L2 Error (f - f_c_bar)')
            # plt.xlabel('Rank')
            # plt.ylabel('Error')
            # plt.title('Error Convergence')
            # plt.legend()
            # plt.xscale('log')
            
            reconstructed_values_2d_c_r = np.array([
            reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_fix]), np.array([v_j, vy_fix]), C_app, D)
            for x_i, v_j in zip(XX.flatten(), VV.flatten())
        ]).reshape(XX.shape)

            reconstructed_values_2d_c_r_als = np.array([
            reconstructed_maxwellian(phi_funcs, psi_funcs, np.array([x_i, y_fix]), np.array([v_j, vy_fix]), C_app_als, D)
            for x_i, v_j in zip(XX.flatten(), VV.flatten())
        ]).reshape(XX.shape)
##########################################################################            
            
        elif D == 3:
            analytical_values_1d = [
                analytical_maxwellian(rho_3d(x_i, y_i,z_i), D, u=[u_13d(x_i), u_23d(y_i), u_33d(z_i)], T=T_3d(x_i, y_i, z_i), ve=np.array([v_x, v_y, v_z]))
                for x_i, y_i, z_i, v_x, v_y, v_z in zip(x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid)]
            
            reconstructed_values_1d =  [reconstructed_maxwellian(phi_basis_x, psi_basis_v, np.array([x_i, y_i, z_i]), np.array([v_x, v_y, v_z]), C, D) for x_i, y_i, z_i, v_x, v_y, v_z in zip(x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid)]
            
            analytical_array = np.array(analytical_values_1d)
            reconstructed_array = np.array(reconstructed_values_1d)
            error = analytical_array - reconstructed_array
            n = len(analytical_array)
            dx = (x_a - x_b) / (n - 1)  
            dy = dx
            dvx = dx
            dvy = dx
            # Trapezoidal rule (second-order accuracy)
            weights = np.ones(n)
            weights[0] = weights[-1] = 0.5  # Half weight at endpoints
            error_L2 = np.sqrt(np.sum(error**2) * dx *dy *dz * dvx *dvy *dvz) 
            # L1 norm (absolute error with quadrature)
            error_L1 = dx *dy *dz * dvx *dvy *dvz * np.sum(weights * np.abs(error))
            print(f"L1 error: {error_L1}")
            print(f"L2 error: {error_L2}")
            err_L1.append(error_L1)
            err_L2.append(error_L2)
            
            analytical_values_2d = np.array([
            analytical_maxwellian(rho_3d(x_i, y_fix, z_fix), D, u=[u_13d(x_i), u_23d(y_i), u_33d(z_i)], T=T_3d(x_i, y_fix, z_fix), ve=np.array([v_j, vy_fix, vz_fix]))
            for x_i, v_j in zip(X.flatten(), V.flatten())
        ]).reshape(X.shape)
        
            reconstructed_values_2d = np.array([
            reconstructed_maxwellian(phi_basis_x, psi_basis_v, np.array([x_i, y_fix, z_fix]), np.array([v_j, vy_fix, vz_fix]), C, D)
            for x_i, v_j in zip(X.flatten(), V.flatten())
        ]).reshape(X.shape)
            
           
        
        #Plot 1D slice
        fig, ax = plt.subplots()
        ax.plot(analytical_values_1d, label='Analytical Maxwellian')
        ax.plot(reconstructed_values_1d, label='Reconstructed Maxwellian')
        ax.set_xlabel('positions')
        ax.set_ylabel('Maxwellian values')
        ax.set_title('Maxwellian in reconstructed values and analytical values over positions')
        ax.legend()
        plt.show()
    
        
        # Plot 2D surfaces
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        ax = ax.flatten()
        c1 = ax[0].contourf(XX, VV, analytical_values_2d, levels=50, cmap='viridis')
        c2 = ax[1].contourf(XX, VV, reconstructed_values_2d, levels=50, cmap='viridis')
        c3 = ax[2].contourf(XX, VV, reconstructed_values_2d_c_r, levels=50, cmap='viridis')
        c4 = ax[3].contourf(XX, VV, reconstructed_values_2d_c_r_als, levels=50, cmap='viridis')
        fig.colorbar(c1, ax=ax[0])
        fig.colorbar(c2, ax=ax[1])
        fig.colorbar(c3, ax=ax[2])
        fig.colorbar(c4, ax=ax[3])
        ax[0].set_title('Analytical Maxwellian, (y=v_y=0.0)')
        ax[1].set_title('Reconstructed Maxwellian, (y=v_y=0.0)')
        ax[2].set_title('AMDM Reconstructed Maxwellian, (y=v_y=0.0)')
        ax[3].set_title('ALS Reconstructed Maxwellian, (y=v_y=0.0)')
        plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot()
    plt.plot(sample_sizes, err, label='Error Maxwellian')
    #plt.plot(sample_sizes, err_L2, label='L2 Error Maxwellian')
    #plt.plot(sample_sizes, err_rho, label='Error particle density')
    #plt.plot(sample_sizes, err_L2_rho, label='L2 Error reconstructed particle density')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error')
    plt.title('Error Convergence')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    # plt.subplot(1, 2, 2)
    # plt.plot(sample_sizes, err_L2, label='L2 Error')
    # plt.xlabel('Number of Samples')
    # plt.ylabel('L2 Error')
    # plt.title('L2 Error Convergence')
    # plt.legend()
    # plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
        
##########################################################    
if __name__ == "__main__":
    main()