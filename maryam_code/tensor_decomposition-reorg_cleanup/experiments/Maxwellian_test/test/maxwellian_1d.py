import numpy as np

# Parameters
rho = 1.0         # Density
T = 1.0  # Temperature
u_1d = np.array([0])        # Bulk velocity in 1D
# u_2d = np.array([0, 0])     # Bulk velocity in 2D
# u_3d = np.array([0, 0, 0])  # Bulk velocity in 3D # Bulk velocity 
d_v = len(u_1d)      # Dimensionality of velocity space
num_samples = 10000


# Sample velocities

samples = np.random.normal(u_1d, np.sqrt(T), (num_samples, d_v))

def maxwell_pdf(v):
    return rho / ((2 * np.pi * T)**(d_v/2)) * np.exp(-((v - u_1d[0])**2) / (2 * T))

pdf_values = maxwell_pdf(samples[:, 0])  # Evaluate for all sampled velocities in 1D