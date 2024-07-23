import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

from utilities.get_data import get_data
from utilities.ring_simulation import ring_simulation
from utilities.build_connectivity_matrix import build_connectivity_matrix
from utilities.build_stimulus_input import build_stimulus_input
from utilities.input_values import DEBUG, MAX_NEURON_NUMBER, TOTAL_TIME_STEPS, \
    TAU_M, DT, THETA_NEURONS, INPUT_BIAS, FIRING_RATE, VISUAL_CUE

# Turn on "Interactive Mode" for matplotlib, this lets us plot everything out and 
# lets the code can keep running
plt.ion()

# Get the data from the CSV file
experimental_data = get_data()

# partition out the pieces we want for simuluation, this isn't necessary but it makes me feel better.
estimate_and_motion_direction = experimental_data[['estimate_angle', 'motion_direction', 'motion_coherence']]

# Build the connectivity matrix
connectivity_matrix = build_connectivity_matrix(max_neuron_number=MAX_NEURON_NUMBER,
                                                theta_neurons=THETA_NEURONS)

if DEBUG:
    print("DEBUG PLOTS INCOMING!")
    fig1 = plt.figure(101)
    ax1 = fig1.add_subplot()
    im1 = ax1.imshow(connectivity_matrix, aspect='auto')
    cbar= fig1.colorbar(im1)
    plt.show()

# firing_rate_out, _ = ring_simulation(total_time_steps=TOTAL_TIME_STEPS,
#                     connectivity_matrix=connectivity_matrix,
#                     input_bias=INPUT_BIAS,
#                     firing_rate=FIRING_RATE,
#                     dt=DT,
#                     tau_m=TAU_M)

stimulus_input_bias = build_stimulus_input(firing_rate=FIRING_RATE,
                                           init_input_bias=INPUT_BIAS,
                                           visual_cue=VISUAL_CUE,
                                           motion_coherence=0.12,
                                           k_constant=0.1)

new_firing_rate, percept_decision = ring_simulation(total_time_steps=TOTAL_TIME_STEPS,
                                                    connectivity_matrix=connectivity_matrix,
                                                    input_bias=stimulus_input_bias,
                                                    firing_rate=FIRING_RATE,
                                                    dt=DT,
                                                    tau_m=TAU_M)


if DEBUG:
    print(f"Decision: {np.argmax(new_firing_rate[-1, :])}")

    plt.figure(8800)
    max_index = np.argmax(new_firing_rate[-1,:])
    plt.plot(range(MAX_NEURON_NUMBER), new_firing_rate[-1,:])
    plt.title("Ring Simulation Output")
    plt.text(45, np.max(new_firing_rate[-1,:]), f"Max Neuron: {max_index}", ha='right', va='top')  # Add a text label for the maximum value
    plt.show()

# last line here for a break point
print("Done")