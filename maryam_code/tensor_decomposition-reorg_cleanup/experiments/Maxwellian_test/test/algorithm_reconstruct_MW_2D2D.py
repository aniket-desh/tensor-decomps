import numpy as np
import matplotlib.pyplot as plt

# Spatial density function
def rho(x, y):
    R = 1 # Radius
    if x**2 + y**2 <= R**2:
        return 1.0
    else:
        return 0.1

# Rejection sampling
number_samples = 10 # Number of samples
sampled_positions = []

# Define the domain bounds
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
D = 4 # Area of D = [-1, 1]**2


# Sample (x, y) uniformly from the domain D
x = np.random.uniform(x_min, x_max, number_samples)
y = np.random.uniform(y_min, y_max, number_samples)
positions = np.column_stack((x, y))

# Compute rho(x, y) in the domain D
density_values = np.array([rho(x, y) for (x, y) in positions])
p_max = np.max(density_values) # Maximum of rho(x, y)

for i in range(len(density_values)):
    r = density_values[i] / p_max # Compute r(x, y) (rejection sampling)
    
    # Accept/reject based on r
    if np.random.uniform(0, 1) < r:
        sampled_positions.append(positions[i])
        
num_samples = len(sampled_positions) # Number of samples


# Bulk velocity components
def u_1(y):
    # return 0.0
    v_0 = 0.1
    delta = 1 / 30
    if y <= 0.5:
        return v_0 * np.tanh((y - 0.25) / delta)
    else:
        return v_0 * np.tanh((0.75 - y) / delta)

def u_2(x):
    # return 0.0
    delta = 5e-3  
    return delta * np.sin(2 * np.pi * x)
    
# def u_3(z):
#     return 0.0

# Compute bulk velocity for sampled positions 3D
bulk_velocities = np.array([[u_1(y), u_2(x)] for x, y in sampled_positions])
# bulk_velocities = np.array([[0, 0, 0] for x, y in sampled_positions]) 

# Define temperature 
# def T(x, y):
#     return 1.0
def T(x, y):
    return 1.0 + 0.1 * (x + y) 


# Sample velocities for each position
sampled_velocities = []
for (x, y), (u_x, u_y) in zip(sampled_positions, bulk_velocities):
    mean_velocity = [u_x, u_y]
    variance = T(x, y)
    velocity_sample = np.random.normal(mean_velocity, np.sqrt(variance), size=2)
    sampled_velocities.append(velocity_sample)

sampled_velocities = np.array(sampled_velocities)

# Validate velocity sampling 
err_mean = []
err_var = []
for i in range(num_samples):  
    x_sample, y_sample = sampled_positions[i]
    v_samples = sampled_velocities[i]

    # Compute empirical mean and variance
    empirical_mean = np.mean(v_samples, axis=0)
    empirical_variance = np.var(v_samples, axis=0)

    # Analytical values
    analytical_mean = np.array([u_1(y_sample), u_2(x_sample)])
    analytical_variance = T(x_sample, y_sample)

    err_mean.append(np.abs(empirical_mean - analytical_mean))
    #print(err_mean)
    err_var.append(np.abs(empirical_variance - analytical_variance))
    #print(err_var)

###################################
err_var = np.array(err_var)
err_mean = np.array(err_mean)

fig, ax = plt.subplots()

# Plot the error in variance and mean
ax.plot(err_var, label='Error in Variance')
ax.plot(err_mean, label='Error in Mean')

ax.set_xlabel('Sampled position')
ax.set_ylabel('Error')
ax.set_title('Error in Variance and Mean Over Samples')
ax.legend()
plt.show()
########################################
# Analytical Maxwellian function
def analytical_maxwellian(rho, u, T, v):
    d_v = len(v)  # Dimensionality of velocity space
    return rho / ((2 * np.pi * T)**(d_v / 2)) * np.exp(-np.linalg.norm(v - u)**2 / (2 * T))

# Compute analytical Maxwellian values for comparison
analytical_values = [
    analytical_maxwellian(rho(x,y), u=[u_1(y), u_2(x)], T=T(x, y), v=velocity)
    for (x, y), velocity in zip(sampled_positions, sampled_velocities)
]


# Reconstructed Maxwellian function
# Spatial basis functions (Fourier)
def phi_i(x):
    return [np.sin(2 * np.pi * i * x) for i in range(3)] + [np.cos(2 * np.pi * i * x) for i in range(3)]

# Velocity basis functions (Fourier)
def psi_j(v):
    return [np.sin(2 * np.pi * i * v) for i in range(3)] + [np.cos(2 * np.pi * i * v) for i in range(3)]



num_basis_space = len(phi_i(0))  # Number of spatial basis functions
num_basis_velocity = len(psi_j(0))  # Number of velocity basis functions

# Initialize tensor C
C = np.zeros((num_basis_space, num_basis_space, num_basis_velocity, num_basis_velocity))


# Compute coefficients C_ijklm

for i in range(num_basis_space):
    for j in range(num_basis_space):
        for k in range(num_basis_velocity):
            for l in range(num_basis_velocity):
                    for (x, y), velocity in zip(sampled_positions, sampled_velocities):
                        C[i, j, k, l] += (
                        phi_i(x)[i] *
                        phi_i(y)[j] *
                        psi_j(velocity[0])[k] *  
                        psi_j(velocity[1])[l] 
                    )

# Normalize by the number of samples
C /= num_samples


def reconstructed_maxwellian(x, y, v_x, v_y, C):
    phi_x_values = phi_i(x)
    phi_y_values = phi_i(y)
    psi_vx_values = psi_j(v_x)
    psi_vy_values = psi_j(v_y)

    result = 0
    for i in range(len(phi_x_values)):
        for j in range(len(phi_y_values)):
            for k in range(len(psi_vx_values)):
                for l in range(len(psi_vy_values)):
                        result += (
                            C[i, j, k, l] *
                            phi_x_values[i] *
                            phi_y_values[j] *
                            psi_vx_values[k] *
                            psi_vy_values[l] 
                        )
    return result

# Reconstructed values for comparison
reconstructed_values = [reconstructed_maxwellian(x, y, velocity[0], velocity[1] , C) for (x, y), velocity in zip(sampled_positions, sampled_velocities)]


##################################
# Plot the error reconstructed_values and analytical values of Maxwellian function

fig, ax = plt.subplots()

ax.plot(reconstructed_values, label='Reconstructed values')
ax.plot(analytical_values, label='Analytical values')
ax.set_xlabel('Sampled position')
ax.set_ylabel('Error')
ax.set_title('Error in reconstructed values and analytical values over samples')
ax.legend()
plt.show()



# def rho(x, y):
#     return 1 - (x**2 + y**2) / 2  

# ###############################################
# #  Gaussian Sampling
# num_samples = 1000 # Number of samples
# mu_x, mu_y = 0.0, 0.0  # Mean for x and y
# sigma_x, sigma_y = 0.5, 0.5  # Standard deviation for x and y

# # Sample x and y from Gaussian distribution
# sampled_positions = []
# x_samples = np.random.normal(mu_x, sigma_x, num_samples)
# y_samples = np.random.normal(mu_y, sigma_y, num_samples) 
# sampled_positions = np.column_stack((x_samples, y_samples))
# #print(sampled_positions)
####################################################

# # Number of samples to generate
# num_samples = 1000

# # Rejection sampling
# samples = []
# while len(samples) < num_samples:
#     # Step 1: Sample (x, y) uniformly from the domain D
#     x = np.random.uniform(x_min, x_max)
#     y = np.random.uniform(y_min, y_max)

#     # Step 2: Compute r(x, y)
#     r = p(x, y) / p_max

#     # Step 3: Accept/reject based on r(x, y)
#     if np.random.uniform(0, 1) < r:
#         samples.append([x, y])
# for i in range (num_samples):
#     # Sample uniformly 
#     x = np.random.uniform(-1 , 1)
#     y = np.random.uniform(-1 , 1)

#     # Accept/reject based on rho(x, y)
#     if np.random.uniform(0, 1) < rho(x, y):
#         sampled_positions.append([x, y])
# #print(sampled_positions)
        
# # ######################################################################
# # Create spatial grid
# x_grid = np.linspace(-1, 1, 100)
# y_grid = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x_grid, y_grid)

# # Compute density values
# density_values = np.array([[rho(xi, yi) for xi in x_grid] for yi in y_grid])

# # Normalize densities to form probabilities
# probabilities = density_values / np.sum(density_values)

# # Sample positions based on probabilities
# num_samples = 10
# sampled_indices = np.random.choice(len(X.flatten()), size=num_samples, p=probabilities.flatten())
# sampled_positions = np.column_stack((X.flatten()[sampled_indices], Y.flatten()[sampled_indices]))
# print(sampled_positions)