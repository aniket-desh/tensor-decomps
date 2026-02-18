import numpy as np
from analythical_functions import rho_1d, u_1d, T_1d, rho_2d, u_12d, u_22d, T_2d, rho_3d, u_13d, u_23d, u_33d, T_3d
import random

def rejection_sampling_positions(intervals, number_samples, D):
    #number_samples = 1000 # Number of samples
    sampled_positions = []
    
    if D == 1:
        # Sample x uniformly from the domain 
        positions = np.random.uniform(intervals[0][0], intervals[0][1], number_samples)
        #positions = np.array(positions)
        #print(f"pos1 {positions}")
        # Compute rho(x) in the domain D
        density_values = np.array([rho_1d(x) for x in positions])
    elif D == 2:
        # Sample (x, y) uniformly from the domain D
        x = np.random.uniform(intervals[0][0], intervals[0][1], number_samples)
        y = np.random.uniform(intervals[1][0], intervals[1][1], number_samples)
        positions = np.column_stack((x, y))
        density_values = np.array([rho_2d(x, y) for (x, y) in positions])
        #print(f"pos2 {positions}")
    elif D == 3:
        # Sample (x, y, z) uniformly from the domain D
        x = np.random.uniform(intervals[0][0], intervals[0][1], number_samples)
        y = np.random.uniform(intervals[1][0], intervals[1][1], number_samples)
        z = np.random.uniform(intervals[2][0], intervals[2][1], number_samples)
        positions = np.column_stack((x, y, z)) 
        density_values = np.array([rho_3d(x, y,z) for (x, y, z) in positions])
        #print(f"pos3 {positions}")
    p_max = np.max(density_values) # Maximum of rho(x)
    #p_max = 1
    
    for i in range(len(density_values)):
        r = density_values[i] / p_max # Compute r(x) (rejection sampling)
        
        # Accept/reject based on r
        if np.random.uniform(0, 1) < r:
            sampled_positions.append(positions[i])
            
    num_samples = len(sampled_positions) # Number of samples
    return num_samples, sampled_positions


def sample_velocities(sampled_positions, D):
    if D == 1:
        # Compute bulk velocity for sampled positions 1D
        #print(sampled_positions)
        bulk_velocities = np.array([u_1d(x) for x in sampled_positions]) 
        
        # Sample velocities for each position
        sampled_velocities = []
        for x, u_x in zip(sampled_positions, bulk_velocities):
            mean_velocity = [u_x]
            variance = T_1d(x)
            velocity_sample = np.random.normal(mean_velocity, np.sqrt(variance))
            sampled_velocities.append(velocity_sample)
    elif D == 2:
        #print(sampled_positions)
            # Compute bulk velocity for sampled positions 2D
        bulk_velocities = np.array([[u_12d(y), u_22d(x)] for x, y in sampled_positions])
 
        # Sample velocities for each position
        sampled_velocities = []
        for (x, y), (u_x, u_y) in zip(sampled_positions, bulk_velocities):
            mean_velocity = [u_x, u_y]
            variance = T_2d(x, y)
            velocity_sample = np.random.normal(mean_velocity, np.sqrt(variance), size=2)
            sampled_velocities.append(velocity_sample)
    elif D == 3:
            # Compute bulk velocity for sampled positions 3D
        #print(sampled_positions)
        bulk_velocities = np.array([[u_13d(x), u_23d(y), u_33d(z)] for x, y, z in sampled_positions])
 
        # Sample velocities for each position
        sampled_velocities = []
        for (x, y, z), (u_x, u_y, u_z) in zip(sampled_positions, bulk_velocities):
            mean_velocity = [u_x, u_y, u_z]
            variance = T_3d(x, y, z)
            velocity_sample = np.random.normal(mean_velocity, np.sqrt(variance), size=3)
            sampled_velocities.append(velocity_sample)       
    return sampled_velocities
    
def sample_gaussian_mixture_3d(n_samples, means, variances, weights):
    # Generate samples from a 3D Gaussian mixture model
    
    samples = np.zeros((n_samples, 3))
    
    # Select components based on weights
    component_indices = np.random.choice(len(weights), size=n_samples, p=weights)
    
    for i in range(n_samples):
        idx = component_indices[i]
        # Sample from selected component
        x = random.gauss(means[idx][0], np.sqrt(variances[idx][0]))
        y = random.gauss(means[idx][1], np.sqrt(variances[idx][1]))
        z = random.gauss(means[idx][2], np.sqrt(variances[idx][2]))
        samples[i] = [x, y, z]
    
    return samples    