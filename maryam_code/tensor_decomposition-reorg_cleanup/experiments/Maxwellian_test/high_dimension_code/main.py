import numpy as np
import sympy as sp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import argparse
from scipy.linalg import svd
from basis_functions import generate_b_spline_basis, chebyshev_polynomial, fourier_basis, legendre_polynomial 
from integration import compute_M1_M2, compute_Mi
from sampling import rejection_sampling_positions, sample_velocities
from analythical_functions import rho_1d, u_1d, T_1d, rho_2d, u_12d, u_22d, T_2d, rho_3d, u_13d, u_23d, u_33d, T_3d, analytical_maxwellian, maxwellian_ana
from reconstruction_c_hat_MW import compute_C_hat, reconstructed_maxwellian, reconstructed_maxwellian_ana, svd_decomposition_truncated
from arg_def import parse_arguments
import tensor_decomposition.backend.numpy_ext as tenpy
from scipy.integrate import simpson

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
            phi_basis_x.append(phi_basis)
            M1_i = compute_Mi(phi_basis, degree_spatial[i], symbol_spatial[i], interval_x[i][0], interval_x[i][1])
            M1.append(M1_i)    
    elif args.basis_functions_spatial == "Fourier":
        for i in range(D):
            phi_basis = [fourier_basis(symbol_spatial[i], j) for j in range(degree_spatial[i])]
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
            psi_basis = [chebyshev_polynomial(symbol_v[i], j) for j in range(degree_v[i])]
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)
    elif args.basis_functions_velocity == "Legendre":
        for i in range(D):
            psi_basis = [legendre_polynomial(symbol_v[i], j) for j in range(degree_v[i])]
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)
    elif args.basis_functions_velocity == "Fourier":    
        for i in range(D):
            psi_basis = [fourier_basis(symbol_v[i], j) for j in range(degree_v[i])]
            psi_basis_v.append(psi_basis)
            M2_i = compute_Mi(psi_basis, degree_v[i], symbol_v[i], interval_v[i][0], interval_v[i][1])
            M2.append(M2_i)
################################################################################
    #sample_sizes = [10, 100, 1000, 10000, 100000]
    sample_sizes = [1000000]
    err_L1 = []
    err_L2 = []
    for number_samples in sample_sizes:
        # Sample data 
        _, x_samples = rejection_sampling_positions(interval_x, number_samples, D)
        v_samples = sample_velocities(x_samples, D)
        # x_samples = np.array(x_samples)
        # v_samples = np.array(v_samples)
    
    
        # Compute C_hat 
        C_hat = compute_C_hat(phi_basis_x, psi_basis_v, x_samples, v_samples, D)
        # print(len(C_hat))
        # print(len(M1))
        # print(len(M2))
        # Compute C
        if D == 1:
            #C = np.linalg.inv(M1).T @ C_hat @ np.linalg.inv(M2)
            C = np.einsum('ki,kl,lj->ij', np.linalg.inv(M1[0]), C_hat, np.linalg.inv(M2[0]))
        elif D == 2:
            C = np.einsum('ia,jb,ijkl,kc,ld->abcd', np.linalg.inv(M1[0]), np.linalg.inv(M1[1]), C_hat, np.linalg.inv(M2[0]), np.linalg.inv(M2[1]))
        elif D == 3:
            C = np.einsum('ia,jb,kc,ijklmn,ld,me,nf->abcdef', np.linalg.inv(M1[0]), np.linalg.inv(M1[1]), np.linalg.inv(M1[2]), C_hat, np.linalg.inv(M2[0]), np.linalg.inv(M2[1]),  np.linalg.inv(M2[2])) 
            
        #print(C)
        # print(M1)
        # print(M2)
        # print(C_hat)
        # print(C)
        
        ##########################################
        # #test reconstructed and analythical values 
        # x_grid = np.linspace(x_a, x_b, x_grid_size)
        # v_grid = np.linspace(v_a, v_b, v_grid_size)
        
        # analytical_values = [
        #     analytical_maxwellian(rho(x_i), u=[u(x_i)], T=T(x_i), ve=velocity)
        #     for x_i, velocity in zip(x_grid, v_grid)]
        
        # reconstructed_values =  [reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C) for x_i, v_j in zip(x_grid, v_grid)]
        
        # # Compute the error norm \\M_true - M_reconstructed\\_2
        # analytical_array = np.array(analytical_values)
        # reconstructed_array = np.array(reconstructed_values)
        
        # error_norm = np.linalg.norm(analytical_array - reconstructed_array)
        # print(f"Error Norm: {error_norm:}")
        # Integrate over x domain
    
    #     error_function = analytical_array - reconstructed_array
    #     if D == 1:
    #         error_norm_simpson = simpson(error_function**2, x_grid)
    #     elif D == 2:
    #         error_norm_simpson = simpson([simpson(error_row**2, x_grid) for error_row in error_function.reshape(x_grid_size, x_grid_size)], x_grid)
    #     elif D == 3:
    #         error_3d = error_function.reshape(x_grid_size, x_grid_size, x_grid_size)
    #         error_norm_simpson_3d = simpson(
    #     [simpson(
    #         [simpson(error_3d[i, j]**2, x_grid) for j in range(x_grid_size)],
    #         x_grid
    #     ) for i in range(x_grid_size)],
    #     x_grid
    # )
    #     print(f"{D}D Error Norm (Simpson's quadrature): {np.sqrt(error_norm_simpson_3d)}")
        # print(f"Error Norm (Simpson's quadrature): {np.sqrt(error_norm_simpson)}")
        # ###################################################
        # # Plot the error reconstructed_values and analytical values of Maxwellian function
        
        # fig, ax = plt.subplots()
        
        # ax.plot(reconstructed_values, label='Reconstructed values')
        # ax.plot(analytical_values, label='Analytical values')
        # ax.set_xlabel('positions')
        # ax.set_ylabel('Maxwellian values')
        # ax.set_title('Maxwellian in reconstructed values and analytical values over positions')
        # ax.legend()
        # plt.show()
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
        # # Integrate error over x for fixed v
        # x_integrated_error = [integrate_over_x(v_j) for v_j in v_grid]
    
        # # 2. Integrate error over v for fixed x
        # v_integrated_error = [integrate_over_v(x_i) for x_i in x_grid]
        # # error_norm = np.linalg.norm(analytical_values.flatten() - reconstructed_values.flatten())
        # # print(f"Error Norm: {error_norm:}")
    
        # Create 2D grid
        # x_grid = np.linspace(x_a, x_b, x_grid_size)
        # v_grid = np.linspace(v_a, v_b, v_grid_size)
        # X, V = np.meshgrid(x_grid, v_grid)
        
        # # Compute analytical Maxwellian over the grid
        # analytical_values = np.array([
        #     analytical_maxwellian(rho(x_i), u=u(x_i), T=T(x_i), ve=v_j)
        #     for x_i, v_j in zip(X.flatten(), V.flatten())
        # ]).reshape(X.shape)
    ##################################################################
        # # Compute reconstructed Maxwellian over the grid
        # reconstructed_values = np.array([
        #     reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C)
        #     for x_i, v_j in zip(X.flatten(), V.flatten())
        # ]).reshape(X.shape)
    
        # # Plot 2D surfaces
        # fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # # Analytical Maxwellian
        # c1 = ax[0].contourf(X, V, analytical_values, levels=50, cmap='viridis')
        # fig.colorbar(c1, ax=ax[0])
        # ax[0].set_title('Analytical Maxwellian')
        # ax[0].set_xlabel('X')
        # ax[0].set_ylabel('Velocity')
        
        # # Reconstructed Maxwellian
        # c2 = ax[1].contourf(X, V, reconstructed_values, levels=50, cmap='viridis')
        # fig.colorbar(c2, ax=ax[1])
        # ax[1].set_title('Reconstructed Maxwellian')
        # ax[1].set_xlabel('X')
        # ax[1].set_ylabel('Velocity')
        
        # plt.tight_layout()
        # plt.show()
    # [...] (Previous code for basis generation, sampling, and matrix computation)
    #####################################################
        ######################################################################
        # # Generate 2D grid
        # x_grid = np.linspace(x_a, x_b, x_grid_size)
        # v_grid = np.linspace(vx_a, vx_b, v_grid_size)
        # X, V = np.meshgrid(x_grid, v_grid)
        
        # # Compute analytical and reconstructed values over the grid
        # analytical_values_2d = np.array([
        #     analytical_maxwellian(rho_1d(x_i), D, u=u_1d(x_i), T=T_1d(x_i), ve=v_j)
        #     for x_i, v_j in zip(X.flatten(), V.flatten())
        # ]).reshape(X.shape)
        
        # reconstructed_values_2d = np.array([
        #     reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C, D)
        #     for x_i, v_j in zip(X.flatten(), V.flatten())
        # ]).reshape(X.shape)
    ##############################################################
        
        x_grid = np.linspace(x_a, x_b, x_grid_size)
        y_grid = np.linspace(y_a, y_b, x_grid_size)
        z_grid = np.linspace(z_a, z_b, x_grid_size)
        vx_grid = np.linspace(vx_a, vx_b, v_grid_size)
        vy_grid = np.linspace(vy_a, vy_b, v_grid_size)
        vz_grid = np.linspace(vz_a, vz_b, v_grid_size)
        X, V = np.meshgrid(x_grid, vx_grid)
        y_fix = 0.0
        vy_fix = 0.0
        z_fix = 0.0
        zy_fix = 0.0
        
        if D == 1:
            analytical_values_1d = [
                analytical_maxwellian(rho_1d(x_i), D, u=[u_1d(x_i)], T=T_1d(x_i), ve=velocity)
                for x_i, velocity in zip(x_grid, vx_grid)]
           # print(analytical_values_1d)
            reconstructed_values_1d =  [reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C, D) for x_i, v_j in zip(x_grid, vx_grid)]
            
            
            #print(reconstructed_values_1d)
            analytical_array = np.array(analytical_values_1d)
            reconstructed_array = np.array(reconstructed_values_1d)
            error = analytical_array - reconstructed_array
            n = len(analytical_array)
            dx = (x_a - x_b) / (n - 1)  
            dv = (vx_a - vx_b) / (n - 1)
            # Trapezoidal rule (second-order accuracy)
            weights = np.ones(n)
            weights[0] = weights[-1] = 0.5  # Half weight at endpoints
            error_L2 = np.sqrt(np.sum(error**2) * dx * dv)
            # L1 norm (absolute error with quadrature)
            error_L1 = dx * dv* np.sum(weights * np.abs(error))
            print(f"L1 error: {error_L1}")
            print(f"L2 error: {error_L2}")
            err_L1.append(error_L1)
            err_L2.append(error_L2)
########################################################################
            U, sigma, VT = np.linalg.svd(C, full_matrices=False)
            #print(C)
            print(sigma)
            max_rank = min(C.shape)
            ranks = list(range(1, max_rank + 1))
            errors1_L1 = []
            errors2_L1 = []
            for r in ranks:
                C_r = svd_decomposition_truncated(U, sigma, VT, r)
                if r == 3:
                    C_app = C_r 
                reconstructed_values_1d_C_r =  [reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C_r, D) for x_i, v_j in zip(x_grid, vx_grid)]
                reconstructed_array_C_r = np.array(reconstructed_values_1d_C_r)
                error1 = analytical_array - reconstructed_array_C_r
                error2 = reconstructed_array - reconstructed_array_C_r

                errors1_L1.append(dx * dv* np.sum(weights * np.abs(error1)))
                errors2_L1.append(dx * dv* np.sum(weights * np.abs(error2)))
            plt.figure(figsize=(10, 5))
            plt.subplot()
            plt.plot(ranks, errors1_L1, label='L1 Error (M - f_c_bar)')
            plt.plot(ranks, errors2_L1, label='L1 Error (f - f_c_bar)')
            plt.xlabel('Rank')
            plt.ylabel('Error')
            plt.title('Error Convergence')
            plt.legend()
            plt.xscale('log')
###############################################################################                
            analytical_values_2d = np.array([
            analytical_maxwellian(rho_1d(x_i), D, u=u_1d(x_i), T=T_1d(x_i), ve=v_j)
            for x_i, v_j in zip(X.flatten(), V.flatten())
        ]).reshape(X.shape)
            #print(analytical_values_2d)
            reconstructed_values_2d = np.array([
            reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C, D)
            for x_i, v_j in zip(X.flatten(), V.flatten())]).reshape(X.shape)
            #print(f"re:{reconstructed_values_2d}")
            
            reconstructed_values_2d_c_r = np.array([
            reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C_app, D)
            for x_i, v_j in zip(X.flatten(), V.flatten())]).reshape(X.shape)
        
        elif D == 2:
            analytical_values_1d = [
                analytical_maxwellian(rho_2d(x_i, y_i), D, u=[u_12d(y_i), u_22d(x_i)], T=T_2d(x_i, y_i), ve=np.array([v_x, v_y]))
                for x_i, y_i, v_x, v_y in zip(x_grid, y_grid, vx_grid, vy_grid)]
            
            reconstructed_values_1d =  [reconstructed_maxwellian(phi_basis_x, psi_basis_v, np.array([x_i, y_i]), np.array([v_x, v_y]), C, D) for x_i, y_i, v_x, v_y in zip(x_grid, y_grid, vx_grid, vy_grid)]
            

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
            error_L2 = np.sqrt(np.sum(error**2) * dx *dy * dvx *dvy)
            # L1 norm (absolute error with quadrature)
            error_L1 = dx *dy * dvx *dvy * np.sum(weights * np.abs(error))
            print(f"L1 error: {error_L1}")
            print(f"L2 error: {error_L2}")
            err_L1.append(error_L1)
            err_L2.append(error_L2)
#########################################################################
            I, J, K, L = C.shape 
            R = [5, 10, 20, 30, 50, 100]
            C_unfold = C.reshape(I* J, K, L)
            C_folded = C_unfold.reshape(I, J, K, L)
##########################################################################            
            analytical_values_2d = np.array([
            analytical_maxwellian(rho_2d(x_i, y_fix), D, u=[u_12d(y_fix), u_22d(x_i)], T=T_2d(x_i, y_fix), ve=np.array([v_j, vy_fix]))
            for x_i, v_j in zip(X.flatten(), V.flatten())
        ]).reshape(X.shape)
        
            reconstructed_values_2d = np.array([
            reconstructed_maxwellian(phi_basis_x, psi_basis_v, np.array([x_i, y_fix]), np.array([v_j, vy_fix]), C, D)
            for x_i, v_j in zip(X.flatten(), V.flatten())
        ]).reshape(X.shape)
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
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        c1 = ax[0].contourf(X, V, analytical_values_2d, levels=50, cmap='viridis')
        c2 = ax[1].contourf(X, V, reconstructed_values_2d, levels=50, cmap='viridis')
        c3 = ax[2].contourf(X, V, reconstructed_values_2d_c_r, levels=50, cmap='viridis')
        fig.colorbar(c1, ax=ax[0])
        fig.colorbar(c2, ax=ax[1])
        fig.colorbar(c3, ax=ax[2])
        ax[0].set_title('Analytical Maxwellian, (y=v_y=0.0)')
        ax[1].set_title('Reconstructed Maxwellian, (y=v_y=0.0)')
        ax[2].set_title('Approximated Reconstructed Maxwellian, (y=v_y=0.0)')
        plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot()
    plt.plot(sample_sizes, err_L1, label='L1 Error')
    plt.plot(sample_sizes, err_L2, label='L2 Error')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error')
    plt.title('Error Convergence')
    plt.legend()
    plt.xscale('log')
    
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