import numpy as np

# Parameters
rho = 1.0         # Density 
T = 1.0           # Temperature
u = np.array([0, 0, 0])  # Bulk velocity in 3D 
d_v = len(u)      # Dimensionality of velocity space 
num_samples = 10
sigma = np.sqrt(T)

# Generate samples for each velocity component (v_x, v_y, v_z)
v_x = np.random.normal(u[0], sigma, num_samples)
v_y = np.random.normal(u[1], sigma, num_samples)
v_z = np.random.normal(u[2], sigma, num_samples)

# Compute speeds (magnitude of velocity vectors)
speeds = np.sqrt(v_x**2 + v_y**2 + v_z**2)

# Analytical Maxwellian speed PDF in 3D
def maxwell_speed_pdf(v):
    return 4 * np.pi * v**2 * (1 / (2 * np.pi * T))**(3/2) * np.exp(-v**2 / (2 * T))
pdf_values = maxwell_pdf(speeds[:, 0])  # Evaluate for all sampled velocities in 1D
print(pdf_values)
# Evaluate analytical PDF 
bins = 8
speed_grid = np.linspace(0, max(speeds), bins)
analytical_pdf = maxwell_speed_pdf(speed_grid)


# Spatial grid (2D)
x_grid = np.linspace(0, 1, 10)
y_grid = np.linspace(0, 1, 10)

# Velocity grid (3D)
v_x_grid = np.linspace(-2 * sigma, 2 * sigma, 10)
v_y_grid = np.linspace(-2 * sigma, 2 * sigma, 10)
v_z_grid = np.linspace(-2 * sigma, 2 * sigma, 10)

# Initialize tensor
tensor_shape = (len(x_grid), len(y_grid), len(v_x_grid), len(v_y_grid), len(v_z_grid))
space_velocity_tensor = np.zeros(tensor_shape)


# for i in range(len(x_grid)):
#     for j in range(len(y_grid)):
#         space_velocity_tensor[i, j] = maxwell_speed_pdf(v_x_grids)

mean_speed = np.mean(speeds)
variance_speed_squared = np.var(speeds**2)

print("Mean Speed:", mean_speed)
print("Expected Mean Speed:", np.sqrt(8 * T / np.pi))
print("Variance of Speed Squared:", variance_speed_squared)
print("Expected Variance of Speed Squared:", 3 * T)

