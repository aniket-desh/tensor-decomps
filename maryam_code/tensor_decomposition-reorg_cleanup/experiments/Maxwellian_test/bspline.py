import sympy as sp
import numpy as np
import argparse
from integration import compute_integral
from sympy.functions.special.bsplines import bspline_basis

# symbolic variables
x = sp.symbols('x')
v = sp.symbols('v')

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