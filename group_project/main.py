import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import logging

from utilities.get_data import get_data
from utilities.ring_simulation import ring_simulation
from utilities.build_connectivity_matrix import build_connectivity_matrix
from utilities.build_stimulus_input import build_stimulus_input
from utilities.circular_error import circular_error
from utilities.input_values import DEBUG, MAX_NEURON_NUMBER, TOTAL_TIME_STEPS, \
    TAU_M, DT, THETA_NEURONS, INPUT_BIAS, FIRING_RATE, STIMULUS_BUMP_WIDTH, K_CONSTANT, SUBJECTS_LIST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Turn on "Interactive Mode" for matplotlib, this lets us plot everything out and 
# lets the code can keep running
if DEBUG:
    plt.ion()

# Set random seed for reproducibility
np.random.seed(1001)

# Get the data from the CSV file
experimental_data = get_data()

subject_filtered_data = experimental_data[experimental_data['subject_id'].isin(SUBJECTS_LIST)]

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

def process_row(index, row):
    estimate_angle = row['estimate_angle']
    visual_cue = row['motion_direction']
    motion_coherence = row['motion_coherence']

    stimulus_input_bias = build_stimulus_input(firing_rate=FIRING_RATE,
                                               init_input_bias=INPUT_BIAS,
                                               visual_cue=visual_cue,
                                               motion_coherence=motion_coherence,
                                               stimulus_bump_width=STIMULUS_BUMP_WIDTH,
                                               k_constant=K_CONSTANT)

    new_firing_rate, percept_decision = ring_simulation(total_time_steps=TOTAL_TIME_STEPS,
                                                        connectivity_matrix=connectivity_matrix,
                                                        input_bias=stimulus_input_bias,
                                                        firing_rate=FIRING_RATE,
                                                        dt=DT,
                                                        tau_m=TAU_M)
    
    simulation_error = circular_error(estimate_angle, percept_decision)
    
    if index % 100 == 0:
        logging.info(f"Subject: {row['subject_id']}, Motion Coherence: {motion_coherence}, Decision: {percept_decision}, Error: {simulation_error}")
    
    return index, percept_decision, simulation_error

# Use ThreadPoolExecutor to parallelize the computation
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_row, index, row) for index, row in subject_filtered_data.iterrows()]
    for future in concurrent.futures.as_completed(futures):
        index, percept_decision, simulation_error = future.result()
        subject_filtered_data.loc[index, "percept_decision"] = percept_decision
        subject_filtered_data.loc[index, "simulation_error"] = simulation_error

# Save the data to a CSV file
subject_filtered_data.to_csv("subject_filtered_data.csv", index=False)

# last line here for a break point
logging.info("SIMULATION COMPLETE")