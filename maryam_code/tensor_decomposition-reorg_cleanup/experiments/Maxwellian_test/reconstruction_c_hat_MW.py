import numpy as np
import sympy as sp

# symbolic variables
x = sp.symbols('x')
v = sp.symbols('v')

def compute_C_hat(phi_basis, psi_basis, x_samples, v_samples):

    #Compute C_hat[k, l] = (1/N) \sum \phi_k(x_sample)\psi_l(v_sample).

    num_phi = len(phi_basis)
    num_psi = len(psi_basis)
    C_hat = np.zeros((num_phi, num_psi))
    
    # Lambdify converts SymPy's symbolic B-spline basis functions into numerical functions
    phi_funcs = [sp.lambdify(x, phi, 'numpy') for phi in phi_basis]
    psi_funcs = [sp.lambdify(v, psi, 'numpy') for psi in psi_basis]
    
    for k in range(num_phi):
        for l in range(num_psi):
            for x_i, v_i in zip(x_samples, v_samples):
                C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    return C_hat / len(x_samples)
    
def reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C):
    
    num_phi = len(phi_basis_x)
    num_psi = len(psi_basis_v)
    result = 0

    # # Lambdify converts SymPy's symbolic B-spline basis functions into numerical functions
    # phi_funcs = [sp.lambdify(x, phi, 'numpy') for phi in phi_basis_x]
    # psi_funcs = [sp.lambdify(v, psi, 'numpy') for psi in psi_basis_v]
    
    for i in range((num_phi)):
        for j in range((num_psi)):
            result += (
                    C[i, j] *
                    phi_funcs[i](x_i) * 
                    psi_funcs[j](v_j) 
                        )
    return result
def reconstructed_maxwellian_ana(phi_basis_x, psi_basis_v, C):
    
    num_phi = len(phi_basis_x)
    num_psi = len(psi_basis_v)
    result = 0

    for i in range((num_phi)):
        for j in range((num_psi)):
            result += (
                    C[i, j] *
                    phi_basis_x[i] * 
                    psi_basis_v[j] 
                        )
    return result   
# def compute_C_hat_basis(phi_basis, psi_basis, x_samples, v_samples):

#     #Compute C_hat[k, l] = (1/N) \sum \phi_k(x_sample)\psi_l(v_sample).

#     num_phi = len(phi_basis)
#     num_psi = len(psi_basis)
#     C_hat = np.zeros((num_phi, num_psi))
    
#     for k in range(num_phi):
#         for l in range(num_psi):
#             for x_i, v_i in zip(x_samples, v_samples):
#                 C_hat[k, l] += phi_basis(x_i)[k] * psi_basis(v_i)[l]
    
#     return C_hat / len(x_samples)

    
# def reconstructed_maxwellian_basis(phi_basis_x, psi_basis_v, x_i, v_j, C):
    
#     num_phi = len(phi_basis_x)
#     num_psi = len(psi_basis_v)
#     result = 0

    
#     for i in range((num_phi)):
#         for j in range((num_psi)):
#             result += (
#                     C[i, j] *
#                     phi_basis_x(x_i)[i] * 
#                     psi_basis_v(v_j)[j] 
#                         )
#     return result   
    
