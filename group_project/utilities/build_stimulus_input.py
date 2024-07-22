import numpy as np
import matplotlib.pyplot as plt
from utilities.input_values import DEBUG, CONSTANT_INPUT_BIAS, MAX_MOTION_COHERENCE, RANDOM_BASELINE_ADJUSTMENT

def build_stimulus_input(k_constant: float,
                         motion_coherence: float,
                         firing_rate: np.ndarray,
                         init_input_bias: np.ndarray,
                         visual_cue: int = 225):

    

    firing_rate[0,:]=firing_rate[-1,:]
    input_bias = np.random.random(size=len(init_input_bias)) - RANDOM_BASELINE_ADJUSTMENT

    for i in range(5):
        input_bias[int(visual_cue-4+i)] += (motion_coherence/MAX_MOTION_COHERENCE)*k_constant*(1+i)*CONSTANT_INPUT_BIAS
        input_bias[int(visual_cue+i)] += (motion_coherence/MAX_MOTION_COHERENCE)*k_constant*(5-i)*CONSTANT_INPUT_BIAS

    if DEBUG:
        plt.figure(5500)
        plt.plot(input_bias)
        plt.show()

    return input_bias