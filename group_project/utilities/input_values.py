import numpy as np

DEBUG = False

# Initialising all the variables

MAX_NEURON_NUMBER = 360
TOTAL_TIME_STEPS = 500
CONSTANT_INPUT_BIAS = 0.1
TAU_M = 0.1 #membrane time constant
DT = 0.005 #time step for the simulation
TOTAL_TIME_ELAPSED = TOTAL_TIME_STEPS * DT

THETA_NEURONS = np.array([(2*np.pi*i)/MAX_NEURON_NUMBER for i in range(MAX_NEURON_NUMBER)])
INPUT_BIAS = np.ones(MAX_NEURON_NUMBER)*CONSTANT_INPUT_BIAS

#array of size (neuron_number,length of simulation time steps)
FIRING_RATE = np.random.rand(TOTAL_TIME_STEPS,len(THETA_NEURONS))*0.01 #initialise all the voltages at the first time step arbitrarily

# Vars in building the visual stimulus
VISUAL_CUE = 225
K_CONSTANT = 0.03
RANDOM_BASELINE_ADJUSTMENT = 0.5
MAX_MOTION_COHERENCE = 0.24 # 0.24 is the maximum motion coherence value in the data
STIMULUS_BUMP_WIDTH = 10

# Simulation information 
SUBJECTS_LIST = [11] # 4, 7, 9, 11 <- all good subject numbers. Running just 11 for now.

# Variables in the optimization function
INITIAL_GUESS = 1.0