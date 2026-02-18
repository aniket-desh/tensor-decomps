import numpy as np
import sympy as sp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import argparse
from scipy.linalg import svd
from basis_functions import generate_b_spline_basis, chebyshev_polynomial, fourier_basis, legendre_polynomial, legendre_polynomial_scaled, chebyshev_polynomial_scaled, gaussian_basis
from integration import compute_M1_M2, compute_Mi
from sampling import rejection_sampling_positions, sample_velocities, sample_gaussian_mixture_3d
from analythical_functions import rho_1d, u_1d, T_1d, rho_2d, u_12d, u_22d, T_2d, rho_3d, u_13d, u_23d, u_33d, T_3d, analytical_maxwellian, maxwellian_ana, gaussian_mixture_3d_pdf
from reconstruction_c_hat_MW import compute_C_hat, reconstructed_maxwellian, reconstructed_maxwellian_ana, svd_decomposition_truncated, compute_C_hat_Gau, reconstructed_Gaussian
from arg_def import parse_arguments
import tensor_decomposition.backend.numpy_ext as tenpy
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
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

    
    parser = argparse.ArgumentParser()
    #arg_def.parse_arguments(parser)
    args = parse_arguments()
#####################################################################################
    arg_defs.add_general_arguments(parser)
    arg_defs.add_sparse_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_amdm_arguments(parser)
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
    #Dimentionality of spatial and velocity
    D = args.D
    means = np.array(args.means).reshape(D, 3)
    variances = np.array(args.variances).reshape(D, 3)
    weights = np.array(args.weights)
    
    args.tlib = "numpy"  # Force use of NumPy backend

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
    
    # Sampling parameters
    number_samples = args.number_samples

    # Basis functions
    basis_functions_spatial = args.basis_functions_spatial
    
    # Grid parameters
    x_grid_size = args.x_grid_size

    # Generate spatial and velocity grids
    x_grid = np.linspace(x_a, x_b, x_grid_size)
    y_grid = np.linspace(y_a, y_b, x_grid_size)
    z_grid = np.linspace(z_a, z_b, x_grid_size)
    
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
    
################################################################################
    #sample_sizes = [10, 10000, 1000000]
    sample_sizes = [1000000]
    #sample_sizes = [1000000]
    err_L1 = []
    err_L2 = []
    err_L1_rho = []
    err_L2_rho = []
    err = []
    phi_funcs = []
    for d in range(D):
        phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis_x[d]]
        phi_funcs.append(phi_fun)
        
    for number_samples in sample_sizes:
        # Sample data 
        x_samples = sample_gaussian_mixture_3d(number_samples, means, variances, weights)
        # x_samples = np.array(x_samples)
        # v_samples = np.array(v_samples)
    
    
        # Compute C_hat 
        C_hat = compute_C_hat_Gau(phi_basis_x, x_samples, D)
        #print((C_hat))
        #print((M1))

        # Compute C
            #C = np.linalg.inv(M1).T @ C_hat @ np.linalg.inv(M2)
        C = np.einsum('ia,jb,kc,ijk->abc', np.linalg.inv(M1[0]), np.linalg.inv(M1[1]), np.linalg.inv(M1[2]), C_hat) 
            
        
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
        
        if D == 3:
            X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
            analytical_values_1d = np.array([
                gaussian_mixture_3d_pdf(x_i, y_i, z_i, means, variances, weights) for x_i, y_i, z_i in zip(X.flatten(), Y.flatten(), Z.flatten())]).reshape(X.shape)
           # print(analytical_values_1d)
            reconstructed_values_1d = np.array([reconstructed_Gaussian(phi_funcs, np.array([x_i, y_i, z_i]), C, D) for x_i, y_i, z_i in zip(X.flatten(), Y.flatten(), Z.flatten())]).reshape(X.shape)
            
            analytical_array = analytical_values_1d
            reconstructed_array = reconstructed_values_1d
            #error = np.linalg.norm(analytical_array - reconstructed_array)
            error = (np.linalg.norm(analytical_array - reconstructed_array))/np.sqrt(len(analytical_array.flatten()))
            err.append(error)
            print(f"error: {error}")
            # error = analytical_array - reconstructed_array
            # dx = x_grid[1] - x_grid[0]  
            # dy = y_grid[1] - y_grid[0]  
            # dz = z_grid[1] - z_grid[0]  
            
            # # Trapezoidal weights for each dimension
            # weights_x = np.ones(len(x_grid))
            # weights_x[0] = weights_x[-1] = 0.5  
            # weights_y = np.ones(len(y_grid))
            # weights_y[0] = weights_y[-1] = 0.5  
            # weights_z = np.ones(len(z_grid))
            # weights_z[0] = weights_z[-1] = 0.5   
            
            # weights_3d = np.einsum('i,j,k->ijk', weights_x, weights_y, weights_z)
            
            # # L1 error (integrate absolute error )
            # error_L1 = np.sum(weights_3d * np.abs(error)) * dx * dy * dz
            
            # # L2 error (integrate squared error over 3D grid)
            # error_L2 = np.sqrt(np.sum(weights_3d * error**2) * dx * dy *dz)
            # print(f"L1 error: {error_L1}")
            # print(f"L2 error: {error_L2}")
            # err_L1.append(error_L1)
            # err_L2.append(error_L2)
        
#########################################################################
         
            A_ini = generate_initial_guess(tenpy, C, args)
            #C_folded = C_unfold.reshape(I, J, K, L)
            #ranks = [5, 10, 20, 30, 50, 100]
            ranks = [10]
            errors1_L1 = []
            errors2_L1 = []
            O = None 
            for R in ranks:
                #print(args)
                
                B, res_als = CP_ALS(tenpy, A_ini, C, O, num_iter, csv_file=None, Regu=None, method='DT', args=args, res_calc_freq=1, tol=1e-05)

                A_ini = generate_initial_guess(tenpy, C, args)
                A, res_amdm = CP_Mahalanobis(tenpy, A_ini, C, O, num_iter, csv_file=None, Regu=None, args=args, res_calc_freq=1)
                C_r_unfold = tenpy.einsum('ir, jr, kr-> ijk', *A)
                C_r = C_r_unfold
                C_r_unfold_als = tenpy.einsum('ir, jr, kr-> ijk', *B)
                C_r_als = C_r_unfold_als
                if R == 10:
                    C_app = C_r 
                    C_app_als = C_r_als

                reconstructed_values_1d_C_r =  np.array([reconstructed_Gaussian(phi_funcs, np.array([x_i, y_i, z_i]), C_r, D) for x_i, y_i, z_i in zip(X.flatten(), Y.flatten(), Z.flatten())]).reshape(X.shape)
            

                reconstructed_values_1d_C_r_als =  np.array([reconstructed_Gaussian(phi_funcs, np.array([x_i, y_i, z_i]), C_r_als, D) for x_i, y_i, z_i in zip(X.flatten(), Y.flatten(), Z.flatten())]).reshape(X.shape)
            
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
            
        
##########################################################################                     
           
        
        #Plot 1D slice
        # fig, ax = plt.subplots()
        # ax.plot(analytical_values_1d, label='Analytical Gaussian')
        # ax.plot(reconstructed_values_1d, label='Reconstructed Gaussian')
        # ax.set_xlabel('positions')
        # ax.set_ylabel('Gaussian values')
        # ax.set_title('Gaussian in reconstructed values and analytical values over positions')
        # ax.legend()
        # plt.show()


        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        y_slice = len(y_grid) // 2
        X2D, Z2D = np.meshgrid(x_grid, z_grid)
        surf = ax.plot_surface(X2D, Z2D, analytical_values_1d[:, y_slice, :], cmap='viridis')
        plt.colorbar(surf)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X2D, Z2D, reconstructed_values_1d[:, y_slice, :], cmap='viridis')
        plt.colorbar(surf)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X2D, Z2D, reconstructed_values_1d_C_r[:, y_slice, :], cmap='viridis')
        plt.colorbar(surf)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X2D, Z2D, reconstructed_values_1d_C_r_als[:, y_slice, :], cmap='viridis')
        plt.colorbar(surf)
        
        # Plot 2D surfaces
        fig, axs = plt.subplots(4, 3, figsize=(12, 8))
        
        # Select middle slices for each dimension
        x_mid = len(x_grid) // 2
        y_mid = len(y_grid) // 2
        z_mid = len(z_grid) // 2
        
        # Analytical Gaussian projections
        # XY projection (fixed z)
        c1 = axs[0, 0].contourf(X[:, :, z_mid], Y[:, :, z_mid], 
                               analytical_values_1d[:, :, z_mid], 
                               levels=50, cmap='viridis')
        axs[0, 0].set_title('Analytical: XY Projection')
        axs[0, 0].set_xlabel('X')
        axs[0, 0].set_ylabel('Y')
        
        # XZ projection (fixed y)
        c2 = axs[0, 1].contourf(X[:, y_mid, :], Z[:, y_mid, :], 
                               analytical_values_1d[:, y_mid, :], 
                               levels=50, cmap='viridis')
        axs[0, 1].set_title('Analytical: XZ Projection')
        axs[0, 1].set_xlabel('X')
        axs[0, 1].set_ylabel('Z')
        
        # YZ projection (fixed x)
        c3 = axs[0, 2].contourf(Y[x_mid, :, :], Z[x_mid, :, :], 
                               analytical_values_1d[x_mid, :, :], 
                               levels=50, cmap='viridis')
        axs[0, 2].set_title('Analytical: YZ Projection')
        axs[0, 2].set_xlabel('Y')
        axs[0, 2].set_ylabel('Z')
        
        # Reconstructed Gaussian projections
        # XY projection
        c4 = axs[1, 0].contourf(X[:, :, z_mid], Y[:, :, z_mid], 
                               reconstructed_values_1d[:, :, z_mid], 
                               levels=50, cmap='viridis')
        axs[1, 0].set_title('Reconstructed: XY Projection')
        axs[1, 0].set_xlabel('X')
        axs[1, 0].set_ylabel('Y')
        
        # XZ projection
        c5 = axs[1, 1].contourf(X[:, y_mid, :], Z[:, y_mid, :], 
                               reconstructed_values_1d[:, y_mid, :], 
                               levels=50, cmap='viridis')
        axs[1, 1].set_title('Reconstructed: XZ Projection')
        axs[1, 1].set_xlabel('X')
        axs[1, 1].set_ylabel('Z')
        
        # YZ projection
        c6 = axs[1, 2].contourf(Y[x_mid, :, :], Z[x_mid, :, :], 
                               reconstructed_values_1d[x_mid, :, :], 
                               levels=50, cmap='viridis')
        axs[1, 2].set_title('Reconstructed AMDM: YZ Projection')
        axs[1, 2].set_xlabel('Y')
        axs[1, 2].set_ylabel('Z')

         # Approximated Reconstructed Gaussian projections
        # XY projection
        c7 = axs[2, 0].contourf(X[:, :, z_mid], Y[:, :, z_mid], 
                               reconstructed_values_1d_C_r[:, :, z_mid], 
                               levels=50, cmap='viridis')
        axs[2, 0].set_title('Reconstructed AMDM: XY Projection')
        axs[2, 0].set_xlabel('X')
        axs[2, 0].set_ylabel('Y')
        
        # XZ projection
        c8 = axs[2, 1].contourf(X[:, y_mid, :], Z[:, y_mid, :], 
                               reconstructed_values_1d_C_r[:, y_mid, :], 
                               levels=50, cmap='viridis')
        axs[2, 1].set_title('Reconstructed AMDM: XZ Projection')
        axs[2, 1].set_xlabel('X')
        axs[2, 1].set_ylabel('Z')
        
        # YZ projection
        c9 = axs[2, 2].contourf(Y[x_mid, :, :], Z[x_mid, :, :], 
                               reconstructed_values_1d_C_r[x_mid, :, :], 
                               levels=50, cmap='viridis')
        axs[2, 2].set_title('Reconstructed AMDM: YZ Projection')
        axs[2, 2].set_xlabel('Y')
        axs[2, 2].set_ylabel('Z')

        # Approximated Reconstructed Gaussian projections
        # XY projection
        c10 = axs[3, 0].contourf(X[:, :, z_mid], Y[:, :, z_mid], 
                               reconstructed_values_1d_C_r_als[:, :, z_mid], 
                               levels=50, cmap='viridis')
        axs[3, 0].set_title('Reconstructed ALS: XY Projection')
        axs[3, 0].set_xlabel('X')
        axs[3, 0].set_ylabel('Y')
        
        # XZ projection
        c11 = axs[3, 1].contourf(X[:, y_mid, :], Z[:, y_mid, :], 
                               reconstructed_values_1d_C_r_als[:, y_mid, :], 
                               levels=50, cmap='viridis')
        axs[3, 1].set_title('Reconstructed ALS: XZ Projection')
        axs[3, 1].set_xlabel('X')
        axs[3, 1].set_ylabel('Z')
        
        # YZ projection
        c12 = axs[3, 2].contourf(Y[x_mid, :, :], Z[x_mid, :, :], 
                               reconstructed_values_1d_C_r_als[x_mid, :, :], 
                               levels=50, cmap='viridis')
        axs[3, 2].set_title('Reconstructed ALS: YZ Projection')
        axs[3, 2].set_xlabel('Y')
        axs[3, 2].set_ylabel('Z')
        
        # Add colorbars
        for i in range(4):
            for j in range(3):
                plt.colorbar(eval(f'c{i*3+j+1}'), ax=axs[i, j])
        
        plt.tight_layout()
        plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot()
    plt.plot(sample_sizes, err, label='Error reconstructed Gaussian')
    #plt.plot(sample_sizes, err_L2, label='L2 Error reconstructed Gaussian')
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
    # # 3D Plotting 
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # Extract X, Y, Z coordinates
    # x = x_samples[:, 0]
    # y = x_samples[:, 1]
    # z = x_samples[:, 2]
    
    # # Plot as a 3D scatter plot
    # ax.scatter(x, y, z, c='skyblue', alpha=0.5, s=10)
    
    # # Formatting
    # ax.set_title('3D Gaussian Mixture Sampling')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # plt.tight_layout()
    # plt.show()
    
##########################################################    
if __name__ == "__main__":
    main()