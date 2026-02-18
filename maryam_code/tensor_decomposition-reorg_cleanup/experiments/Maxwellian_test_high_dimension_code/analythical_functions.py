import numpy as np
import sympy as sp

# Spatial density function
def rho_1d(x):
    R = 0.1  # Radius
    if abs(x) <= R:
        return 2.0
    else:
        return (1 - 2 * 0.2) / 1.8

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
        