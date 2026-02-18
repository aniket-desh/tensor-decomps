import numpy as np
import sympy as sp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import argparse

from bspline import generate_b_spline_basis
from integration import compute_M1_M2, compute_Mi
from sampling import rejection_sampling_positions, sample_velocities
from analythical_functions import rho, u, T, analytical_maxwellian, maxwellian_ana
from reconstruction_c_hat_MW import compute_C_hat, reconstructed_maxwellian, reconstructed_maxwellian_ana
from arg_def import parse_arguments

def main():



    # symbolic variables
    x = sp.symbols('x')
    v = sp.symbols('v')

    
    parser = argparse.ArgumentParser()
    #arg_def.parse_arguments(parser)
    args = parse_arguments()
    
    # Spatial domain parameters
    x_a = args.x_a
    x_b = args.x_b
    num_internal_knots_x = args.num_internal_knots_x
    degree_x = args.degree_x

    # Velocity domain parameters
    v_a = args.v_a
    v_b = args.v_b
    num_internal_knots_v = args.num_internal_knots_v
    degree_v = args.degree_v

    # Sampling parameters
    number_samples = args.number_samples
    
    # Grid parameters
    x_grid_size = args.x_grid_size
    v_grid_size = args.v_grid_size

    # Generate spatial and velocity grids
    x_grid = np.linspace(x_a, x_b, x_grid_size)
    v_grid = np.linspace(v_a, v_b, v_grid_size)
    
############### B-Spline ##############################################    
    Generate spatial and velocity B-spline basis
    phi_basis_x, knots_x = generate_b_spline_basis(x_a, x_b, num_internal_knots_x, degree_x, x)
    psi_basis_v, knots_v = generate_b_spline_basis(v_a, v_b, num_internal_knots_v, degree_v, v)
    
    # Compute M1 (spaatial matrix)and M2 (velocity matrix) symbolically
    M1 = compute_M1_M2(phi_basis_x, knots_x, degree_x, x)
    M2 = compute_M1_M2(psi_basis_v, knots_v, degree_v, v)
    # print(M1)
    # print(M2)
# ############## Chebyshev  #############################################
    # def chebyshev_polynomial(x, n):
    #     if n == 0:
    #         return 1
    #     elif n == 1:
    #         return x
    #     else:
    #         return 2 * x * chebyshev_polynomial(x, n - 1) - chebyshev_polynomial(x, n - 2)

    # phi_basis_x = [chebyshev_polynomial(x, i) for i in range(degree_x)]
    # psi_basis_v = [chebyshev_polynomial(v, j) for j in range(degree_v)]

    # # phi_basis_x_num = [lambda x, i=i: chebyshev_polynomial(x, i) for i in range(degree_x)]
    # # psi_basis_v_num = [lambda v, j=j: chebyshev_polynomial(v, j) for j in range(degree_v)]
    # # #print(phi_basis_x_num)

    # # Compute M1 (spaatial matrix)and M2 (velocity matrix) symbolically
    # M1 = compute_Mi(phi_basis_x, degree_x, x, x_a, x_b)
    # M2 = compute_Mi(psi_basis_v, degree_v, v, v_a, v_b)
    # #print(phi_basis_x_sym)
    # print(M1)
    # print(M2)
###################### Fourier ###################################################    
    # #Spatial basis functions 
    # def fourier_basis_x(x, degree):
    #     return 1j * sp.sin(2 * sp.pi * degree * x) + sp.cos(2 * sp.pi * degree * x) 
    
    # # Velocity basis functions
    # def fourier_basis_v(v, degree):
    #     return 1j * sp.sin(2 * sp.pi * degree * v) + sp.cos(2 * sp.pi * degree * v)
    
    # phi_basis_x = [fourier_basis_x(x, i) for i in range(degree_x)]
    # psi_basis_v = [fourier_basis_v(v, j) for j in range(degree_v)]
    
    # # Compute M1 (spaatial matrix)and M2 (velocity matrix) symbolically
    # M1 = compute_Mi(phi_basis_x, degree_x, x, x_a, x_b)
    # M2 = compute_Mi(psi_basis_v, degree_v, v, v_a, v_b)
####################### Legendre ###############################################
    # def legendre_polynomial(x, n):
    #     # Normalize Legendre polynomials to make them orthonormal
    #     norm = np.sqrt(2 / (2 * n + 1))
    #     if n == 0:
    #         return norm * np.ones_like(x)
    #     elif n == 1:
    #         return norm * x
    #     else:
    #         return norm * ((2 * n - 1) * x * legendre_polynomial(x, n - 1) - (n - 1) * legendre_polynomial(x, n - 2)) / n
    # phi_basis_x = [legendre_polynomial(x, i) for i in range(degree_x)]
    # psi_basis_v = [legendre_polynomial(v, j) for j in range(degree_v)]

    # # Compute M1 (spaatial matrix)and M2 (velocity matrix) symbolically
    # M1 = compute_Mi(phi_basis_x, degree_x, x, x_a, x_b)
    # M2 = compute_Mi(psi_basis_v, degree_v, v, v_a, v_b)
    # #print(phi_basis_x_sym)
    # # print(M1)
    # # print(M2)
################################################################################
    # Sample data 
    _, x_samples = rejection_sampling_positions(x_a, x_b, number_samples)
    v_samples = sample_velocities(x_samples)
    # x_samples = np.array(x_samples)
    # v_samples = np.array(v_samples)


    # Compute C_hat 
    C_hat = compute_C_hat(phi_basis_x, psi_basis_v, x_samples, v_samples)
    
    # Compute C
    C = np.linalg.inv(M1).T @ C_hat @ np.linalg.inv(M2)
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
    ###################################################
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
    # ana_MW = maxwellian_ana(d_v, rho_a, u_a, T_a, v)
    # rec_MW = reconstructed_maxwellian_ana(phi_basis_x, psi_basis_v, C)
    # #print(f"ana_MW: {ana_MW}")
    # #print(f"rec_MW: {rec_MW}")
    # error = sp.Abs(ana_MW - rec_MW)
    # #print(error)
    # # Convert symbolic expression to a numerical function
    # numerical_integrand = sp.lambdify((x, v), error, modules='numpy')

    # #  SciPy's dblquad
    # result, _ = dblquad(
    #     lambda v, x: numerical_integrand(x, v),  
    #     x_a, x_b,                   # x_bound
    #     lambda _: v_a,           # v lower_bound
    #     lambda _: v_b,            # v upper_bound
          #epsabs=1.49e-08,
          #epsrel=1.49e-08
    # )
    # print(f"error: {result:}")
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
    # Generate 2D grid
    x_grid = np.linspace(x_a, x_b, x_grid_size)
    v_grid = np.linspace(v_a, v_b, v_grid_size)
    X, V = np.meshgrid(x_grid, v_grid)
    
    # Compute analytical and reconstructed values over the grid
    analytical_values_2d = np.array([
        analytical_maxwellian(rho(x_i), u=u(x_i), T=T(x_i), ve=v_j)
        for x_i, v_j in zip(X.flatten(), V.flatten())
    ]).reshape(X.shape)
    
    reconstructed_values_2d = np.array([
        reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C)
        for x_i, v_j in zip(X.flatten(), V.flatten())
    ]).reshape(X.shape)
##############################################################
    # Generate 1D slice (fixed position)
    x_grid = np.linspace(x_a, x_b, x_grid_size)
    v_grid = np.linspace(v_a, v_b, v_grid_size)
    
    analytical_values_1d = [
        analytical_maxwellian(rho(x_i), u=[u(x_i)], T=T(x_i), ve=velocity)
        for x_i, velocity in zip(x_grid, v_grid)]
    
    reconstructed_values_1d =  [reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C) for x_i, v_j in zip(x_grid, v_grid)]

    # Error
    analytical_array = np.array(analytical_values_1d)
    reconstructed_array = np.array(reconstructed_values_1d)
    
    error_norm = np.linalg.norm(analytical_array - reconstructed_array)
    print(f"Error Norm: {error_norm:}")
    
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
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    c1 = ax[0].contourf(X, V, analytical_values_2d, levels=50, cmap='viridis')
    c2 = ax[1].contourf(X, V, reconstructed_values_2d, levels=50, cmap='viridis')
    fig.colorbar(c1, ax=ax[0])
    fig.colorbar(c2, ax=ax[1])
    ax[0].set_title('Analytical Maxwellian')
    ax[1].set_title('Reconstructed Maxwellian')
    plt.show()
##########################################################    
if __name__ == "__main__":
    main()