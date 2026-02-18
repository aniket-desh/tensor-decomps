import numpy as np

# Parameters
rho = 1.0          # Density
T = 1.0            # Temperature
u = np.array([0, 0, 0])  # Bulk velocity in 3D
num_samples = 10000
sigma = np.sqrt(T)

# Sample velocities for a fixed spatial position
v_x = np.random.normal(u[0], sigma, num_samples)
v_y = np.random.normal(u[1], sigma, num_samples)
v_z = np.random.normal(u[2], sigma, num_samples)

# Combine into velocity vectors
velocity_samples = np.column_stack((v_x, v_y, v_z))