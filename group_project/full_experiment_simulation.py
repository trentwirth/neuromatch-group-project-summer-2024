import numpy as np
from scipy.optimize import minimize
from utilities.input_values import K_CONSTANT, TOTAL_TIME_STEPS, INPUT_BIAS, FIRING_RATE, DT, TAU_M, MAX_NEURON_NUMBER, THETA_NEURONS
from utilities.ring_simulation import ring_simulation
from utilities.build_stimulus_input import build_stimulus_input
from utilities.get_data import get_data  
from utilities.build_connectivity_matrix import build_connectivity_matrix

# Load experimental data
experimental_data = get_data()
estimate_and_motion_direction = experimental_data[['estimate_angle', 'motion_direction']]

# Build the connectivity matrix
connectivity_matrix = build_connectivity_matrix(max_neuron_number=MAX_NEURON_NUMBER,
                                                theta_neurons=THETA_NEURONS)

def circular_error(estimate_angle, percept_decision):
    # Define the circular error calculation here
    error = np.abs((estimate_angle - percept_decision + 180) % 360 - 180)
    return error

def optimization_function(K_CONSTANT, visual_cue, estimate_angle):
    # Run the simulation with the given K_CONSTANT and visual_cue
    # You might need to adjust the simulation function to accept K_CONSTANT if it affects the simulation
    stimulus_input_bias = build_stimulus_input(firing_rate=FIRING_RATE, init_input_bias=INPUT_BIAS, visual_cue=visual_cue)
    _, percept_decision = ring_simulation(total_time_steps=TOTAL_TIME_STEPS, connectivity_matrix=connectivity_matrix, input_bias=stimulus_input_bias, firing_rate=FIRING_RATE, dt=DT, tau_m=TAU_M)
    return circular_error(estimate_angle, percept_decision)

def main():
    results = []
    for i, row in estimate_and_motion_direction.iterrows():
        visual_cue = row['motion_direction']
        estimate_angle = row['estimate_angle']
        best_fit = minimize(optimization_function, x0=K_CONSTANT, args=(visual_cue, estimate_angle), bounds=[(0.01, 0.2)])
        results.append((visual_cue, best_fit.x, best_fit.fun))
    return results

if __name__ == "__main__":
    optimization_results = main()
    print(optimization_results)