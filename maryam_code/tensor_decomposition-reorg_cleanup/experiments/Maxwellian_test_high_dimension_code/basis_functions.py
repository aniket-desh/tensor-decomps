import sympy as sp
import numpy as np
import argparse
from integration import compute_integral
from sympy.functions.special.bsplines import bspline_basis
from sympy import symbols, chebyshevt
from sympy import legendre, symbols, sqrt

# symbolic variables
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
vx = sp.symbols('vx')
vy = sp.symbols('vy')
vz = sp.symbols('vz')

def generate_knot_vector(a, b, num_internal_knots, degree):
    
    #Generate clamped knot vector 
    
    # Interior knots uniformly
    interior_knots = np.linspace(a, b, num_internal_knots + 2)[1:-1].tolist()

    # Clamped knots at start and end
    start_knots = [a] * (degree + 1)
    end_knots = [b] * (degree + 1)
    
    # Combine boundary and interior knots
    knot_vector = tuple(start_knots + interior_knots + end_knots)
     #knot_vector = tuple(np.linspace(a, b, num_internal_knots))
    #print(knot_vector)
    return knot_vector

def generate_b_spline_basis(a, b, num_internal_knots, degree, symbol):
    
    # Create clamped knots (repeated degree + 1 times at endpoints)
    clamped_knots = generate_knot_vector(a, b, num_internal_knots, degree)
    print(f"number of knots in variable {symbol} is: {len(clamped_knots)}")
    num_basis = len(clamped_knots) - degree - 1
    print(f"number of B-spline basis functions of degree {degree} in variable {symbol} is : {num_basis}")
    basis_functions = []
    
    for i in range(num_basis):
        basis = bspline_basis(degree, clamped_knots, i, symbol)
        norm = normalize_b_spline_basis(basis, clamped_knots, i, degree, symbol)
        basis = basis / norm
        basis_functions.append(basis)
    
    return basis_functions, clamped_knots
def normalize_b_spline_basis(basis, knots, i, degree, symbol):
    norm = compute_integral(basis, basis, symbol, knots[i], knots[i + degree + 1])
    norm = np.sqrt(float(norm))
    return norm    
    

def chebyshev_polynomial(x, n):
    Tn = chebyshevt(n, x)
    if n == 0: #orthonormalization
        return Tn / np.sqrt(np.pi)
    else:
        return Tn * np.sqrt(2/np.pi)

# def chebyshev_polynomial(x, n):
#     if n == 0:
#        T = 1.0
#     elif n == 1:
#        T = x
#     else:
#        T = 2 * x * chebyshev_polynomial(x, n - 1) - chebyshev_polynomial(x, n - 2)
#     if n == 0: # Normalize Chebyshev polynomials to make them orthonormal
#         return T / np.sqrt(np.pi)  
#     else:
#         return T * np.sqrt(2 / np.pi)
def legendre_polynomial(x, n):        
    P = legendre(n, x)
    # Calculate the normalization factor
    norm = sqrt((2 * n + 1) / 2)
    # Normalize the polynomial
    P_normalized = norm * P
    
    # # Convert to a function that can be evaluated at numerical values
    # P_func = lambdify(x, P_normalized, 'numpy')
    
    return P_normalized

# def legendre_polynomial(x, n):
#     # Normalize Legendre polynomials to make them orthonormal
#     if n == 0:
#         P = np.ones_like(x)
#     elif n == 1:
#         P = x
#     else:
#         P = ((2*n-1)*x*legendre_polynomial(x, n-1) - (n-1)*legendre_polynomial(x, n-2))/n
#     norm = np.sqrt((2 * n + 1) / 2)
#     return norm * P

def fourier_basis(x, degree):
    return 1j * sp.sin(2 * sp.pi * degree * x) + sp.cos(2 * sp.pi * degree * x) 

# Chebyshev (Weight: 1/√(1-x²))
# def compute_Mi_chebyshev(basis, symbol, a, b):
#     M = np.zeros((len(basis), len(basis)))
#     for i in range(len(basis)):
#         for j in range(len(basis)):
#             integrand = (basis[i] * basis[j]) / sp.sqrt(1 - symbol**2)
#             M[i, j] = sp.integrate(integrand, (symbol, a, b)).evalf()
#     return M

# # Legendre (Weight: 1)
# def compute_Mi_legendre(basis, symbol, a, b):
#     M = np.zeros((len(basis), len(basis)))
#     for i in range(len(basis)):
#         for j in range(len(basis)):
#             M[i, j] = sp.integrate(basis[i] * basis[j], (symbol, a, b)).evalf()
#     return M

# # Fourier (Orthogonal over [a, b] with periodicity)
# def compute_Mi_fourier(basis, symbol, a, b):
#     M = np.zeros((len(basis), len(basis)))
#     L = b - a
#     for i in range(len(basis)):
#         for j in range(len(basis)):
#             M[i, j] = sp.integrate(basis[i] * basis[j], (symbol, a, b)).evalf()
#     return M
      