import numpy as np

# Parameters
rho = 1.0          # Density
T = 1.0            # Temperature
u = np.array([0, 0, 0])  # Bulk velocity in 3D
d_x = 2            # Spatial dimensions
d_v = len(u)       # Velocity dimensions
num_samples = 10000

# Define spatial grid (e.g., 2D space)
x_grid = np.linspace(0, 1, 50)
y_grid = np.linspace(0, 1, 50)
spatial_positions = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, d_x)

# Sample velocities for each spatial position
sigma = np.sqrt(T)
velocity_samples = [] # Sampled velocities for each spatial position

for pos in spatial_positions:
    # Sample velocities from Gaussian centered at bulk velocity `u`
    v_x = np.random.normal(u[0], sigma, num_samples)
    v_y = np.random.normal(u[1], sigma, num_samples)
    v_z = np.random.normal(u[2], sigma, num_samples)
    velocity_samples.append(np.column_stack((v_x, v_y, v_z)))

# Reconstruct density (rho), bulk velocity (u), and temperature (T)
def compute_moments(samples):
    rho_estimated = len(samples)
    u_estimated = np.mean(samples, axis=0)
    T_estimated = np.mean(np.linalg.norm(samples - u_estimated, axis=1)**2) / d_v
    return rho_estimated, u_estimated, T_estimated

# Compute moments for a single spatial position
rho_sampled, u_sampled, T_sampled = compute_moments(velocity_samples[0])
print("Density:", rho_sampled)
print("Bulk Velocity:", u_sampled)
print("Temperature:", T_sampled)


import numpy as np

# Parameters
T = 1.0  # Temperature
u = np.array([0, 0, 0])  # Bulk velocity in 3D

# Example density function (smooth)
def rho(x):
    return np.sin(np.pi * x[0]) + np.cos(np.pi * x[1])  # Example smooth function

# Define the Maxwellian distribution
def maxwellian(x, v):
    return rho(x) / (2 * np.pi * T)**(3/2) * np.exp(-np.linalg.norm(v - u)**2 / (2 * T))

# Evaluate the Maxwellian at a specific position and velocity
x = np.array([0.5, 0.5])  # Position
v = np.array([1, 1, 1])  # Velocity
print(maxwellian(x, v))

