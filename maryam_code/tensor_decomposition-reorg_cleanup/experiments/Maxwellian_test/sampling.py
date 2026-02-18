import numpy as np
from analythical_functions import rho, u, T

def rejection_sampling_positions(x_min, x_max, number_samples):
    #number_samples = 1000 # Number of samples
    sampled_positions = []
    
    # Define the domain bounds
    #x_min, x_max = -1.0, 1.0
    
    
    # Sample x uniformly from the domain 
    positions = np.random.uniform(x_min, x_max, number_samples)
    # Compute rho(x) in the domain D
    density_values = np.array([rho(x) for x in positions])
    p_max = np.max(density_values) # Maximum of rho(x)
    #p_max = 1
    
    for i in range(len(density_values)):
        r = density_values[i] / p_max # Compute r(x) (rejection sampling)
        
        # Accept/reject based on r
        if np.random.uniform(0, 1) < r:
            sampled_positions.append(positions[i])
            
    num_samples = len(sampled_positions) # Number of samples
    return num_samples, sampled_positions


def sample_velocities(sampled_positions):
    # Compute bulk velocity for sampled positions 2D
    bulk_velocities = np.array([u(x) for x in sampled_positions])
    #bulk_velocities = np.array([[0, 0] for x, y in sampled_positions]) 
    
    
    
    # Sample velocities for each position
    sampled_velocities = []
    for x, u_x in zip(sampled_positions, bulk_velocities):
        mean_velocity = [u_x]
        variance = T(x)
        velocity_sample = np.random.normal(mean_velocity, np.sqrt(variance))
        sampled_velocities.append(velocity_sample)
    
    #sampled_velocities = np.array(sampled_velocities)
    return sampled_velocities