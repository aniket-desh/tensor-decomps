import numpy as np
import sympy as sp

# Spatial density function on [-1, 1]
def rho_1d(x):
    R = 0.1  # Radius
    if abs(x) <= R:
        return 2.0
    else:
        return (1 - 2 * 0.2) / 1.8
# Spatial density function on [-2, 2]        
# def rho_1d(x):
#     R = 0.1  # Radius
#     if abs(x) <= R:
#         return 1.11425
#     else:
#         return 0.20451
        

def u_1d(x):
    # return 0.0
    delta = 5e-3  
    return delta * np.sin(2 * np.pi * x)
    # return delta * x
    #return 0.0
    
def T_1d(x):
    return 1 + 0.001 * x  

def rho_2d(x, y):
    R_2 = 0.5
    if x**2 + y**2 <= R_2:
        return .6
    else:
        return 0.0237    
def u_12d(y):
    # return 0.0
    v_0 = 0.1
    delta = 1 / 30
    if y <= 0.5:
        return v_0 * np.tanh((y - 0.25) / delta)
    else:
        return v_0 * np.tanh((0.75 - y) / delta)

def u_22d(x):
    # return 0.0
    delta = 5e-3  
    return delta * np.sin(2 * np.pi * x)
    
# def u_3(z):
#     return 0.0

# Define temperature 
# def T(x, y):
#     return 1.0
def T_2d(x, y):
    return 1.0 + 0.001 * (x + y) 
    
def analytical_maxwellian(rho, d_v, u, T, ve):
    #d_v = len(ve)  # Dimensionality of velocity space
    #d_v = 1
    return rho / ((2 * np.pi * T)**(d_v / 2)) * np.exp(-np.linalg.norm(ve - u)**2 / (2 * T))

def maxwellian_ana(d_v, rho, u, T, v): 
    return rho / ((2 * sp.pi * T)**(d_v / 2)) * sp.exp(-(v - u)**2 / (2 * T))

def analytical_rho(rho, d_v, u, T, symbol, a, b): 
    if d_v == 1:
        integral = sp.integrate(sp.exp(-(symbol - u)**2 / (2 * T)), (symbol, a, b)) 
    else: 
        integral_vx = sp.integrate(sp.exp(-(symbol[0] - u[0])**2 / (2 * T)), (symbol[0], a, b))  
        integral_vy = sp.integrate(sp.exp(-(symbol[1] - u[1])**2 / (2 * T)), (symbol[1], a, b)) 
        integral = integral_vx * integral_vy 
    return rho / ((2 * sp.pi * T)**(d_v / 2)) * integral 
    
def f(a, b):
    return 1 /(b - a)

def rho_3d(x, y,z):
    return 1.0    
def u_13d(x):
    return 0.0
   
def u_23d(y):
    return 0.0
    
def u_33d(z):
    return 0.0

def T_3d(x, y, z):
    return 1.0 
    
def gaussian_1d(x, mu, var):
    # Compute 1D Gaussian PDF
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-((x - mu) ** 2) / (2 * var))

def gaussian_3d(x, y, z, mu_x, mu_y, mu_z, var_x, var_y, var_z):
    # Compute 3D Gaussian PDF 
    return (
        gaussian_1d(x, mu_x, var_x) *
        gaussian_1d(y, mu_y, var_y) *
        gaussian_1d(z, mu_z, var_z)
    )
def gaussian_mixture_3d_pdf(x, y, z, means, variances, weights):

    # Compute PDF of a 3D Gaussian mixture model
    
    result = 0.0
    for i in range(len(weights)):
        den = gaussian_3d(
            x, y, z, 
            means[i][0], means[i][1], means[i][2],
            variances[i][0], variances[i][1], variances[i][2]
        )
        result += weights[i] * den
    return result


    
# def gaussian_mixture_3d_pdf(x, y, z, means, covariances, weights):
#     """
#     Compute PDF of a 3D Gaussian mixture model.
    
#     Parameters:
#     - x, y, z: Coordinates where to evaluate the PDF
#     - means: List of [mu_x, mu_y, mu_z] for each component
#     - covariances: List of 3x3 covariance matrices for each component
#     - weights: List of weights for each component (should sum to 1)
    
#     Returns:
#     - Probability density at (x, y, z)
#     """
#     result = 0.0
#     for i in range(len(weights)):
#         # Create point vector
#         point = np.array([x, y, z])
        
#         # Compute multivariate normal density
#         diff = point - means[i]
#         inv_cov = np.linalg.inv(covariances[i])
#         det_cov = np.linalg.det(covariances[i])
        
#         exponent = -0.5 * np.dot(np.dot(diff, inv_cov), diff)
#         coefficient = 1 / (np.sqrt((2 * np.pi)**3 * det_cov))
        
#         result += weights[i] * coefficient * np.exp(exponent)
    
#     return result        