import numpy as np


# sample positions
# Define spatial density function rho(x)
def rho(x):
     return np.exp(-np.sum(x**2, axis=1))

# Discretize spatial domain
x_grid = np.linspace(-5, 5, 100)
y_grid = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.column_stack((X.flatten(), Y.flatten()))
densities = rho(positions)

# Normalize densities to form a probability distribution
probabilities = densities / np.sum(densities)

# Sample positions based on rho(x)
num_samples = 10000
sampled_positions_indices = np.random.choice(len(positions), size=num_samples, p=probabilities)
sampled_positions = positions[sampled_positions_indices]



# Sample velocities
# Define bulk velocity u(x) and temperature T(x)
def u(x):
    return np.array([0, 0])  # Example: Constant bulk velocity

def T(x):
    return 1.0  # Example: Constant temperature

# Sample velocities for each position
sampled_velocities = []
for x in sampled_positions:
    mean_velocity = u(x)
    variance = T(x)
    velocity_sample = np.random.normal(loc=mean_velocity, scale=np.sqrt(variance), size=3)
    sampled_velocities.append(velocity_sample)

sampled_velocities = np.array(sampled_velocities)



# compute coefficients 
# Define basis functions (e.g., Fourier or polynomial basis)
def phi_i(x):
    return [1, x[0], x[1]]  # Example: Polynomial basis in space

def psi_j(v):
    return [1, v]  # Example: Linear basis in velocity

# Compute coefficients C_ijkℓ using Monte Carlo sampling
num_basis_functions_x = len(phi_i([0, 0]))
num_basis_functions_v = len(psi_j(0))
C = np.zeros((num_basis_functions_x, num_basis_functions_v, num_basis_functions_v))

for pos, vel in zip(sampled_positions, sampled_velocities):
    phi_values = phi_i(pos)
    psi_values_vx = psi_j(vel[0])
    psi_values_vy = psi_j(vel[1])
    psi_values_vz = psi_j(vel[2])
    
    for i in range(num_basis_functions_x):
        for j in range(num_basis_functions_v):
            for k in range(num_basis_functions_v):
                for l in range(num_basis_functions_v):
                    C[i, j, k] += phi_values[i] * psi_values_vx[j] * psi_values_vy[k] * psi_values_vz[l]

C /= num_samples


# Validate velocity sampling
for i in range(5):  # Check for a few random positions
    x_sample = sampled_positions[i]
    v_samples = sampled_velocities[i]

    # Compute empirical mean and variance
    empirical_mean = np.mean(v_samples, axis=0)
    empirical_variance = np.var(v_samples, axis=0)

    # Compare with analytical values
    analytical_mean = u(x_sample)
    analytical_variance = T(x_sample)

    print(f"Position {x_sample}:")
    print(f"Empirical Mean: {empirical_mean}, Analytical Mean: {analytical_mean}")
    print(f"Empirical Variance: {empirical_variance}, Analytical Variance: {analytical_variance}")


from sklearn.metrics import mean_squared_error

# Evaluate reconstructed Maxwellian
def reconstructed_maxwellian(x, v):
    return sum(
        C[i, j, k] * phi_i(x)[i] * psi_j(v[0])[j] * psi_j(v[1])[k]
        for i in range(len(phi_i([0])))
        for j in range(len(psi_j(0)))
        for k in range(len(psi_j(0)))
    )

# Evaluate true Maxwellian
def true_maxwellian(x, v):
    return rho(x) / (2 * np.pi * T(x)) ** (3 / 2) * np.exp(-np.linalg.norm(v - u(x)) ** 2 / (2 * T(x)))

# Grid of points for validation
x_grid = np.linspace(-5, 5, 10)
v_grid = np.linspace(-5, 5, 10)
X, Vx, Vy = np.meshgrid(x_grid, v_grid, v_grid)
points_x = np.column_stack((X.flatten(), Vy.flatten()))
points_v = np.column_stack((Vx.flatten(), Vy.flatten()))

# Compute reconstructed and true values
reconstructed_values = [reconstructed_maxwellian(px, pv) for px, pv in zip(points_x, points_v)]
true_values = [true_maxwellian(px, pv) for px, pv in zip(points_x, points_v)]

# Compute MSE
mse = mean_squared_error(true_values, reconstructed_values)
print(f"Mean Squared Error between reconstructed and true Maxwellian: {mse}")
#######################################################
import numpy as np

# Define spatial density function
def rho(x, y):
    R = 1.0  # Radius
    if x**2 + y**2 <= R**2:
        return 1.0
    else:
        return 0.1

# Create spatial grid
x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Compute density values
density_values = np.array([[rho(xi, yi) for xi in x_grid] for yi in y_grid])

# Normalize densities to form probabilities
probabilities = density_values / np.sum(density_values)

# Sample positions based on probabilities
num_samples = 10000
sampled_indices = np.random.choice(len(X.flatten()), size=num_samples, p=probabilities.flatten())
sampled_positions = np.column_stack((X.flatten()[sampled_indices], Y.flatten()[sampled_indices]))

# Define bulk velocity components
def u_1(y):
    v_0 = 0.1
    delta = 1 / 30
    if y <= 0.5:
        return v_0 * np.tanh((y - 0.25) / delta)
    else:
        return v_0 * np.tanh((0.75 - y) / delta)

def u_2(x):
    delta = 5e-3  
    return delta * np.sin(2 * np.pi * x)

# Compute bulk velocity for sampled positions
bulk_velocities = np.array([[u_1(y), u_2(x)] for x, y in sampled_positions])

# Define temperature (constant or spatially varying)
def T(x, y):
    return 1.0

# Sample velocities for each position
sampled_velocities = []
for (x, y), (u_x, u_y) in zip(sampled_positions, bulk_velocities):
    # Define a 3D mean velocity (assuming z-component is zero)
    mean_velocity = [u_x, u_y, 0]
    variance = T(x, y)
    velocity_sample = np.random.normal(loc=mean_velocity, scale=np.sqrt(variance), size=3)
    sampled_velocities.append(velocity_sample)

sampled_velocities = np.array(sampled_velocities)

# Output the first few sampled positions, bulk velocities, and sampled velocities
print(sampled_positions[:5], bulk_velocities[:5], sampled_velocities[:5])

# Compute empirical mean and variance
empirical_mean_vx = np.mean(v_x_samples)
empirical_variance_vx = np.var(v_x_samples)

empirical_mean_vy = np.mean(v_y_samples)
empirical_variance_vy = np.var(v_y_samples)

print(f"Empirical Mean (v_x): {empirical_mean_vx}, Variance (v_x): {empirical_variance_vx}")
print(f"Empirical Mean (v_y): {empirical_mean_vy}, Variance (v_y): {empirical_variance_vy}")

# Analytical mean and variance
analytical_mean_vx = 0  # Expected mean of u_1(y) over uniform sampling
analytical_variance_vx = T(0, 0)  # Assuming constant temperature

analytical_mean_vy = 0  # Expected mean of u_2(x) over uniform sampling
analytical_variance_vy = T(0, 0)  # Assuming constant temperature

print(f"Analytical Mean (v_x): {analytical_mean_vx}, Variance (v_x): {analytical_variance_vx}")
print(f"Analytical Mean (v_y): {analytical_mean_vy}, Variance (v_y): {analytical_variance_vy}")



# Analytical Maxwellian function
def analytical_maxwellian(rho, u, T, v):
    d_v = len(v)  # Dimensionality of velocity space
    normalization = rho / ((2 * np.pi * T)**(d_v / 2))
    exponent = -np.linalg.norm(v - u)**2 / (2 * T)
    return normalization * np.exp(exponent)

# Example: Compute analytical Maxwellian at sampled points
analytical_values = [
    analytical_maxwellian(rho=1.0, u=[0, 0], T=1.0, v=vel)
    for vel in velocities  # velocities is a list of sampled velocity vectors
]



# Example basis functions (e.g., polynomial basis)
def phi_i(x):
    return [1, x[0], x[1]]  # Polynomial basis in space

def psi_j(v):
    return [1, v[0], v[1]]  # Polynomial basis in velocity

# Reconstructed Maxwellian function
def reconstructed_maxwellian(coefficients, x, v):
    result = 0
    for i in range(len(phi_i(x))):
        for j in range(len(psi_j(v))):
            result += coefficients[i][j] * phi_i(x)[i] * psi_j(v)[j]
    return result

# Example: Compute reconstructed Maxwellian at sampled points
coefficients = np.random.rand(3, 3)  # Example coefficients (replace with actual values)
reconstructed_values = [
    reconstructed_maxwellian(coefficients, x=[0.5, 0.5], v=vel)
    for vel in velocities
]


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(analytical_values, reconstructed_values)
print(f"Mean Squared Error: {mse}")


relative_errors = np.abs(np.array(analytical_values) - np.array(reconstructed_values)) / np.array(analytical_values)
mean_relative_error = np.mean(relative_errors)
print(f"Mean Relative Error: {mean_relative_error}")


# Convert errors to NumPy arrays for easier manipulation
err_mean = np.array(err_mean)
err_var = np.array(err_var)

# Create an array of indices for the samples
indices = np.arange(num_samples)

# Plot i vs err_mean
plt.figure(figsize=(12, 6))
plt.plot(indices, err_mean[:, 0], label='u_x Mean Error', color='blue', alpha=0.7)
plt.plot(indices, err_mean[:, 1], label='u_y Mean Error', color='green', alpha=0.7)
plt.plot(indices, err_mean[:, 2], label='u_z Mean Error', color='red', alpha=0.7)
plt.title('Sample Index vs Mean Error in Velocity Components')
plt.xlabel('Sample Index (i)')
plt.ylabel('Mean Error')
plt.legend()
plt.grid(True)
plt.show()

# Plot i vs err_var
plt.figure(figsize=(12, 6))
plt.plot(indices, err_var[:, 0], label='u_x Variance Error', color='blue', alpha=0.7)
plt.plot(indices, err_var[:, 1], label='u_y Variance Error', color='green', alpha=0.7)
plt.plot(indices, err_var[:, 2], label='u_z Variance Error', color='red', alpha=0.7)
plt.title('Sample Index vs Variance Error in Velocity Components')
plt.xlabel('Sample Index (i)')
plt.ylabel('Variance Error')
plt.legend()
plt.grid(True)
plt.show()