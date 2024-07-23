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

# because its so much data, fit our constant to the first 6 subjects, test our fit against the last 6
filtered_experimental_data = experimental_data[experimental_data['subject_id'].isin(range(1, 7))]

# grab only the columns we want/need because it makes me happy
estimate_and_motion_direction = filtered_experimental_data[['estimate_angle', 'motion_direction', 'motion_coherence']]

# Build the connectivity matrix
connectivity_matrix = build_connectivity_matrix(max_neuron_number=MAX_NEURON_NUMBER,
                                                theta_neurons=THETA_NEURONS)

def circular_error(estimate_angle, percept_decision):
    error = np.abs((estimate_angle - percept_decision + 180) % 360 - 180)
    return error

def worker(row, input_constant):
    visual_cue = row['motion_direction']
    estimate_angle = row['estimate_angle']
    motion_coherence = row['motion_coherence']

    stimulus_input_bias = build_stimulus_input(k_constant=input_constant,
                                               stimulus_bump_width=10,
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

    logging.info(f"input_constant: {input_constant[0]}, Motion Coherence: {motion_coherence}, Error: {error}")

    return error

def optimization_function(input_constant, estimate_and_motion_direction):
    errors = []
    # Convert DataFrame to a list of dictionaries for easier processing
    rows = estimate_and_motion_direction.to_dict('records')
    
    # Use ProcessPoolExecutor to parallelize the computation
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker, row, input_constant) for row in rows]
        for future in futures:
            errors.append(future.result())
    
    mean_error = np.mean(errors)
    print(f"Input Constant: {input_constant}, Mean Error: {mean_error}")
    return mean_error

def main():
    best_fit = minimize(optimization_function, x0=INITIAL_GUESS, args=(estimate_and_motion_direction,), bounds=[(0.01, 1)])
    print(f"Best fit: {best_fit.x}, Mean Error: {best_fit.fun}")

    return best_fit.x, best_fit.fun

if __name__ == "__main__":

    optimization_results = main()
    print(optimization_results)
    
    # Save the results to a file
    with open('optimization_results.txt', 'w') as f:
        f.write(f"Best fit parameters: {optimization_results[0]}, Mean Error: {optimization_results[1]}")