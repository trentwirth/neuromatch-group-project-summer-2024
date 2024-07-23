import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from utilities.input_values import INITIAL_GUESS, TOTAL_TIME_STEPS, INPUT_BIAS, FIRING_RATE, DT, TAU_M, MAX_NEURON_NUMBER, THETA_NEURONS
from utilities.ring_simulation import ring_simulation
from utilities.build_stimulus_input import build_stimulus_input
from utilities.get_data import get_data  
from utilities.build_connectivity_matrix import build_connectivity_matrix

# Load experimental data
experimental_data = get_data()

# TODO: create parameter matrix for list of subjects
############################################################################################################
SUBJECT_TO_TRAIN_ON = [4]

start_k = 0.1  # Example start value for k_constant
end_k = 1.0    # Example end value for k_constant
num_k = 99     # Number of k_constant values to explore

start_bump = 1  # Example start value for stimulus_bump_width
end_bump = 21    # Example end value for stimulus_bump_width
num_bump = 11     # Number of stimulus_bump_width values to explore

k_constants = np.linspace(start_k, end_k, num_k)
stimulus_bump_widths = np.arange(start_bump, end_bump+1, 2)

# Grab subject to train on
filtered_experimental_data = experimental_data[experimental_data['subject_id'].isin(SUBJECT_TO_TRAIN_ON)]
############################################################################################################

# Initialize results array
results = np.zeros((num_bump, num_k, len(filtered_experimental_data)))

# Build the connectivity matrix
connectivity_matrix = build_connectivity_matrix(max_neuron_number=MAX_NEURON_NUMBER,
                                                theta_neurons=THETA_NEURONS)

def circular_error(estimate_angle, percept_decision):
    error = np.abs((estimate_angle - percept_decision + 180) % 360 - 180)
    return error

def worker(row, input_k_constant, input_stimulus_bump_width):
    visual_cue = row['motion_direction']
    estimate_angle = row['estimate_angle']
    motion_coherence = row['motion_coherence']

    stimulus_input_bias = build_stimulus_input(k_constant=input_k_constant,
                                               stimulus_bump_width=input_stimulus_bump_width,
                                               motion_coherence=motion_coherence, 
                                               firing_rate=FIRING_RATE, 
                                               init_input_bias=INPUT_BIAS, 
                                               visual_cue=visual_cue)
    
    _, percept_decision = ring_simulation(total_time_steps=TOTAL_TIME_STEPS, 
                                          connectivity_matrix=connectivity_matrix, 
                                          input_bias=stimulus_input_bias, 
                                          firing_rate=FIRING_RATE, 
                                          dt=DT, 
                                          tau_m=TAU_M)
    
    error = circular_error(estimate_angle, percept_decision)

    logging.info(f"k_constant: {input_k_constant}, stimulus_bump_width: {input_stimulus_bump_width}, Motion Coherence: {motion_coherence}, Error: {error}")

    return error

# TODO: Create a nested loop structure to iterate over the k_constants and stimulus_bump_widths - allow some parallelization for speed
# Parallel processing function
def process_parameters(k_idx, bump_idx, row_index, row):
    k = k_constants[k_idx]
    bump_width = stimulus_bump_widths[bump_idx]
    error = worker(row, k, bump_width)
    return k_idx, bump_idx, row_index, error

# Use ProcessPoolExecutor to parallelize the computation
with ProcessPoolExecutor() as executor:
    logging.info("Starting parameter search")
    futures = [executor.submit(process_parameters, k_idx, bump_idx, row_index, row.to_dict())
               for k_idx in range(num_k) for bump_idx in range(num_bump) for row_index, row in filtered_experimental_data.iterrows()]
    for future in futures:
        k_idx, bump_idx, row_index, error = future.result()
        results[bump_idx, k_idx, row_index] = error

# Find the minimum error
min_error = np.min(results)
min_error_indices = np.unravel_index(np.argmin(results), results.shape)
best_k = k_constants[min_error_indices[1]]
best_bump_width = stimulus_bump_widths[min_error_indices[0]]

print(f"Best k_constant: {best_k}, Best stimulus_bump_width: {best_bump_width}, Min Error: {min_error}")