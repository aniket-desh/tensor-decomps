import numpy as np
import sympy as sp

# Spatial density function
def rho(x):
    R = 0.1  # Radius
    if abs(x) <= R:
        return 2.0 #4
    else:
        return (1 - 2 * 0.2) / 1.8 #0.1

def u(x):
    # return 0.0
    delta = 5e-3  
    return delta * np.sin(2 * np.pi * x)
    # return delta * x
    #return 0.0
    
def T(x):
    return 1 + 0.001 * x  

def analytical_maxwellian(rho, u, T, ve):
    #d_v = len(ve)  # Dimensionality of velocity space
    d_v = 1
    return rho / ((2 * np.pi * T)**(d_v / 2)) * np.exp(-np.abs(ve - u)**2 / (2 * T))

def maxwellian_ana(d_v, rho, u, T, v):
    # x = sp.symbols('x')
    # v = sp.symbols('v')
    #d_v = len(ve)  # Dimensionality of velocity space
    # d_v = 1
    # rho = 1.0
    # delta = 5e-3  
    # u = delta * sp.sin(2 * sp.pi * x) 
    # T = 1.0 + 0.001 * x  
    return rho / ((2 * sp.pi * T)**(d_v / 2)) * sp.exp(-(v - u)**2 / (2 * T))
    