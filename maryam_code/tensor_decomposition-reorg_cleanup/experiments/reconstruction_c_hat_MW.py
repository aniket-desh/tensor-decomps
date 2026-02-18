import numpy as np
import sympy as sp
from analythical_functions import f
# symbolic variables
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
symbol_spatial = [x, y, z]
vx = sp.symbols('vx')
vy = sp.symbols('vy')
vz = sp.symbols('vz')
symbol_v = [vx, vy, vz]


def compute_C_hat(phi_basis, psi_basis, x_samples, v_samples, D):

    #Compute C_hat[k, l] = (1/N) \sum \phi_k(x_sample)\psi_l(v_sample).
    num_phi_per_dim = [len(phi) for phi in phi_basis] 
    num_psi_per_dim = [len(psi) for psi in psi_basis]
    C_hat = np.zeros(tuple(num_phi_per_dim + num_psi_per_dim))
    
    # Lambdify converts SymPy's symbolic basis functions into numerical functions
    phi_funcs = []
    phi_evals = []
    psi_funcs = []
    psi_evals = []
    # print(f"x_samples {x_samples}")
    # print(f"v_samples {v_samples}")
    # print(phi_basis)
    # print(psi_basis)
    for d in range(D):
        phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis[d]]
        psi_fun = [sp.lambdify(symbol_v[d], psi, 'numpy') for psi in psi_basis[d]]
        phi_funcs.append(phi_fun)
        psi_funcs.append(psi_fun)

        #print(x_samples)
        # print(v_samples)
        if D == 1:
            phi_vals = np.array([[(phi(x_sample)) for phi in phi_fun] for x_sample in x_samples])
            phi_evals.append(phi_vals)
            psi_vals = np.array([[psi(v_sample[0]) for psi in psi_fun] for v_sample in v_samples])
            psi_evals.append(psi_vals) 
        else: 
            phi_vals = np.array([[phi(x_sample[d]) for phi in phi_fun] for x_sample in x_samples])
            phi_evals.append(phi_vals)
            psi_vals = np.array([[psi(v_sample[d]) for psi in psi_fun] for v_sample in v_samples])
            psi_evals.append(psi_vals)       
    # for k in range(num_phi_per_dim):
    #     for l in range(num_psi_per_dim):
    #         for x_i, v_i in zip(x_samples, v_samples):
    #             C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    # return C_hat / len(x_samples)
    # print(f"sample_x {x_samples}")
    # print(f"v_samples {v_samples}")
    # print((phi_evals))
    # print((psi_evals))
    # psi_evals_squeezed = [arr.squeeze() for arr in psi_evals]  # Converts (n, m, 1) → (n, m)
    # #print((psi_evals_squeezed))
    # phi_arr = phi_evals[0]  # Shape (5, 10)
    # psi_arr = psi_evals_squeezed[0]  # Shape (5, 10)
    phi_arr = phi_evals  # Shape (5, 10)
    psi_arr = psi_evals 
    # print((phi_arr))
    # print((psi_arr))
    if D == 1:
        # print((phi_evals))
        # print((psi_evals))
        psi_evals_squeezed = [arr.squeeze() for arr in psi_evals]  # Converts (n, m, 1) → (n, m)
    # #print((psi_evals_squeezed))
        phi_arr = phi_evals[0]  # Shape (5, 10)
        psi_arr = psi_evals_squeezed[0]  # Shape (5, 10)
        C_hat += np.einsum('ni,nj->ij',phi_arr, psi_arr)
    elif D == 2:
        C_hat += np.einsum('ni,nj,nk,nl->ijkl',*phi_evals, *psi_evals)
    elif D == 3:
        C_hat += np.einsum('ni,nj,nk,nl,nm,np->ijklmp',*phi_evals, *psi_evals)
    return C_hat / len(x_samples)    
    
def reconstructed_maxwellian(phi_funcs, psi_funcs, x_i, v_j, C, D):
#Compute result =  \sum C_kl \phi_k(x_i)\psi_l(v_j).
    # print(phi_basis_x)
    # print(psi_basis_v)
    # num_phi_per_dim = [len(phi) for phi in phi_basis_x] 
    # num_psi_per_dim = [len(psi) for psi in psi_basis_v]
    # print(num_phi_per_dim)
    # print(num_psi_per_dim)
    # Lambdify converts SymPy's symbolic basis functions into numerical functions
    
    #phi_funcs = []
    phi_evals = []
    #psi_funcs = []
    psi_evals = []

    for d in range(D):
        # phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis_x[d]]
        # psi_fun = [sp.lambdify(symbol_v[d], psi, 'numpy') for psi in psi_basis_v[d]]
        # phi_funcs.append(phi_fun)
        # psi_funcs.append(psi_fun)
        if D == 1:
            phi_vals = np.array([phi(x_i) for phi in phi_funcs[d]])
            phi_evals.append(phi_vals)
            psi_vals = np.array([psi(v_j) for psi in psi_funcs[d]])
            psi_evals.append(psi_vals) 
        else: 
            # print(x_i)
            # print(v_j)
            phi_vals = np.array([phi(x_i[d]) for phi in phi_funcs[d]])
            phi_evals.append(phi_vals)
            psi_vals = np.array([psi(v_j[d]) for psi in psi_funcs[d]])
            psi_evals.append(psi_vals)       
    # for k in range(num_phi_per_dim):
    #     for l in range(num_psi_per_dim):
    #         for x_i, v_i in zip(x_samples, v_samples):
    #             C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    # return C_hat / len(x_samples)

    #psi_evals_squeezed = [arr.squeeze() for arr in psi_evals]  # Converts (n, m, 1) → (n, m)
    
    result = 0
    if D == 1:
        # print(C)
        psi_evals_squeezed = [arr.squeeze() for arr in psi_evals]  # Converts (n, m, 1) → (n, m)
        phi_arr = phi_evals[0]  
        psi_arr = psi_evals_squeezed[0]  
        result += np.einsum('ij,i,j->',C, phi_arr, psi_arr)
    elif D == 2:
        # print((phi_evals[0]))
        # print((psi_evals[0]))
       
        result += np.einsum('ijkl,i,j,k,l->',C, phi_evals[0], phi_evals[1], psi_evals[0], psi_evals[1])
    elif D == 3:
        result += np.einsum('ijklmn,i,j,k,l,m,n->',C, *phi_evals, *psi_evals)
    return result    
# def reconstructed_maxwellian(phi_basis_x, psi_basis_v, x_i, v_j, C):
    
#     num_phi = len(phi_basis_x)
#     num_psi = len(psi_basis_v)
#     result = 0

#     # Lambdify converts SymPy's symbolic B-spline basis functions into numerical functions
#     phi_funcs = [sp.lambdify(x, phi, 'numpy') for phi in phi_basis_x]
#     psi_funcs = [sp.lambdify(v, psi, 'numpy') for psi in psi_basis_v]
    
#     for i in range((num_phi)):
#         for j in range((num_psi)):
#             result += (
#                     C[i, j] *
#                     phi_funcs[i](x_i) * 
#                     psi_funcs[j](v_j) 
#                         )
#     return result
def compute_C_hat_Gau(phi_basis, x_samples, D):

    #Compute C_hat[k, l] = (1/N) \sum \phi_i(x_sample)\phi_k(z_sample).
    num_phi_per_dim = [len(phi) for phi in phi_basis] 
    C_hat = np.zeros(num_phi_per_dim)
    #print(f"phi_bas {phi_basis}")
    # Lambdify converts SymPy's symbolic basis functions into numerical functions
    phi_funcs = []
    phi_evals = []
    
    # print(f"x_samples {x_samples}")
    for d in range(D):
        phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis[d]]
        #print(f"phi_fun {phi_fun}")
        phi_funcs.append(phi_fun)
        # print(phi_fun)
        # print(psi_fun)
        #print(f"samp: {x_samples}")
        # print(v_samples)
        phi_vals = np.array([[phi(x_sample[d]) for phi in phi_fun] for x_sample in x_samples])
        phi_evals.append(phi_vals)
             
    # for k in range(num_phi_per_dim):
    #     for l in range(num_psi_per_dim):
    #         for x_i, v_i in zip(x_samples, v_samples):
    #             C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    # return C_hat / len(x_samples)
    # print(f"sample_x {x_samples}")
    # print(f"v_samples {v_samples}")
    phi_arr = phi_evals[0]  
    #print(f"phi:{phi_evals}")
    if D == 3:
        C_hat += np.einsum('ni,nj,nk->ijk',*phi_evals)
        #print(C_hat.shape)
    return C_hat / len(x_samples)    
    
def compute_C_hat_constant(phi_basis, x_samples, D):

    #Compute C_hat[k, l] = (1/N) \sum \phi_i(x_sample)
    num_phi_per_dim = [len(phi) for phi in phi_basis] 
    C_hat = np.zeros(num_phi_per_dim)
    #print(f"phi_bas {phi_basis}")
    # Lambdify converts SymPy's symbolic basis functions into numerical functions
    phi_funcs = []
    phi_evals = []
    
    # print(f"x_samples {x_samples}")
    for d in range(D):
        phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis[d]]
        #print(f"phi_fun {phi_fun}")
        phi_funcs.append(phi_fun)
        # print(phi_fun)
        # print(psi_fun)
        #print(f"samp: {x_samples}")
        # print(v_samples)
        phi_vals = np.array([[phi(x_sample) for phi in phi_fun] for x_sample in x_samples])
        phi_evals.append(phi_vals)
             
    # for k in range(num_phi_per_dim):
    #     for l in range(num_psi_per_dim):
    #         for x_i, v_i in zip(x_samples, v_samples):
    #             C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    # return C_hat / len(x_samples)
    # print(f"sample_x {x_samples}")
    # print(f"v_samples {v_samples}")
    phi_arr = phi_evals[0]  
    #print(f"phi:{phi_evals}")
    if D == 1:
        C_hat += np.einsum('ni->i',*phi_evals)
        #print(C_hat.shape)
    return C_hat / len(x_samples)    
        
def reconstructed_Gaussian(phi_funcs, x_i, C, D):
#Compute result =  \sum C_ijk \phi_i(x)\phi_j(y)\phi_k(z).
    #num_phi_per_dim = [len(phi) for phi in phi_basis_x] 
    
    # Lambdify converts SymPy's symbolic basis functions into numerical functions
    #phi_funcs = []
    phi_evals = []

    for d in range(D):
        # phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis_x[d]]
        # phi_funcs.append(phi_fun)
            # print(x_i)
            # print(v_j)
        phi_vals = np.array([phi(x_i[d]) for phi in phi_funcs[d]])
        phi_evals.append(phi_vals)
  
    # for k in range(num_phi_per_dim):
    #     for l in range(num_psi_per_dim):
    #         for x_i, v_i in zip(x_samples, v_samples):
    #             C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    # return C_hat / len(x_samples)

    #psi_evals_squeezed = [arr.squeeze() for arr in psi_evals]  # Converts (n, m, 1) → (n, m)
    
    result = 0
    if D == 3:
        result += np.einsum('ijk,i,j,k->',C, *phi_evals)
    return result  

def reconstructed_constant(phi_basis_x, x_i, C, D):
#Compute result =  \sum C_ijk \phi_i(x)\phi_j(y)\phi_k(z).
    #num_phi_per_dim = [len(phi) for phi in phi_basis_x] 
    
    # Lambdify converts SymPy's symbolic basis functions into numerical functions
    phi_funcs = []
    phi_evals = []
    #print(phi_basis_x)
    for d in range(D):
        phi_fun = [sp.lambdify(symbol_spatial[d], phi, 'numpy') for phi in phi_basis_x[d]]
        phi_funcs.append(phi_fun)
            # print(x_i)
            # print(v_j)
       # print([phi(x_i) for phi in phi_fun])
        phi_vals = np.array([phi(x_i[0]) for phi in phi_fun])
        phi_evals.append(phi_vals)
  
    # for k in range(num_phi_per_dim):
    #     for l in range(num_psi_per_dim):
    #         for x_i, v_i in zip(x_samples, v_samples):
    #             C_hat[k, l] += phi_funcs[k](x_i) * psi_funcs[l](v_i)
    
    # return C_hat / len(x_samples)

    #psi_evals_squeezed = [arr.squeeze() for arr in psi_evals]  # Converts (n, m, 1) → (n, m)
    
    result = 0
    if D == 1:
        result += np.einsum('i,i->',C, *phi_evals)
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
def reconstructed_particle_density(phi_basis_x, psi_basis_v, C, symbol_vx, symbol_vy, a, b, D):
    # print(phi_basis_x[0])
    # print(psi_basis_v[0])
    if D == 1:
        num_phi = len(phi_basis_x[0])
        num_psi = len(psi_basis_v[0])
        result = 0  
        for i in range((num_phi)):
            for j in range((num_psi)):
                integral = sp.integrate(psi_basis_v[0][j], (symbol_vx, a, b))
                result += (
                        C[i, j] *
                        phi_basis_x[0][i] * 
                        integral.evalf() 
                            )
    elif D == 2:
        num_phi_x = len(phi_basis_x[0])
        num_phi_y = len(phi_basis_x[1])
        num_psi_vx = len(psi_basis_v[0])
        num_psi_vy = len(psi_basis_v[1])
        result = 0
    
        for i in range((num_phi_x)):
            for j in range((num_phi_y)):
                for k in range((num_psi_vx)):
                    integral_vx = sp.integrate(psi_basis_v[0][k], (symbol_vx, a, b))
                    for l in range((num_psi_vy)):
                        integral_vy = sp.integrate(psi_basis_v[1][l], (symbol_vy, a, b))
                        result += (
                                C[i, j, k, l] *
                                phi_basis_x[0][i] * 
                                phi_basis_x[1][j] * 
                                integral_vx.evalf() *
                                integral_vy.evalf()
                                    )
    #print(result)    
    return result 
    
def Error(phi_basis_x, C, symbol_spatial, a, b):
    analytical_value = f(a, b)
    reconstructed_expr = sum(C[i] * phi_basis_x[0][i] for i in range(len(C)))
    err = sp.integrate(sp.Abs(analytical_value - reconstructed_expr), (symbol_spatial, a, b))
    
    return err    

def right_hand_side(phi_basis_x, symbol_spatial, a, b):
    f_ana = f(a, b)
    result = []
    for i in range(len(phi_basis_x[0])): 
        integrand = f_ana * phi_basis_x[0][i]
        result.append(sp.integrate(integrand , (symbol_spatial, a, b)))
    return result  
    
def svd_decomposition_truncated(U, sigma, VT, r):
    # Create rank-r approximation
    U_r = U[:, :r]
    sigma_r = sigma[:r]
    VT_r = VT[:r, :]
    C_r = U_r @ np.diag(sigma_r) @ VT_r
    return C_r
     
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
    
