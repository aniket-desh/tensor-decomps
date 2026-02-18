import numpy as np
import sympy as sp
from sympy import re

# symbolic variables
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
vx = sp.symbols('vx')
vy = sp.symbols('vy')
vz = sp.symbols('vz')

def compute_integral(phi_i, phi_j, symbol, a_local, b_local):

    product = phi_i * phi_j
    integral = sp.integrate(product, (symbol, a_local, b_local))
    return integral.evalf()  # Convert to float
    #return float(re(integral).evalf())
def compute_M1_M2(basis, knots, degree, symbol):

    n = len(basis)
    M = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Extract local knots for \phi_i and \phi_j
            support_i = (knots[i], knots[i + degree + 1])
            support_j = (knots[j], knots[j + degree + 1])
            # Calculate overlap interval
            overlap_start = max(support_i[0], support_j[0])
            overlap_end = min(support_i[1], support_j[1])
            
            if overlap_start >= overlap_end:
                M[i, j] = 0.0
            else:
                M[i, j] = compute_integral(basis[i], basis[j], symbol, overlap_start, overlap_end)
    
    return M

    
def compute_Mi(basis, degree, symbol, a, b):

    n = len(basis)
    M = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
                M[i, j] = compute_integral(basis[i], basis[j], symbol, a, b)
    
    return M    