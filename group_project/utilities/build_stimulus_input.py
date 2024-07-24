import numpy as np
import matplotlib.pyplot as plt
from utilities.input_values import DEBUG, MAX_MOTION_COHERENCE, RANDOM_BASELINE_ADJUSTMENT

def build_stimulus_input(firing_rate: np.ndarray,
                         init_input_bias: np.ndarray,
                         visual_cue: int = 225,
                         motion_coherence: float = 0.24,
                         k_constant: float = 1,
                         stimulus_bump_width: int = 5) -> np.ndarray:

    firing_rate[0,:]=firing_rate[-1,:]
    input_bias = np.random.random(size=len(init_input_bias)) - RANDOM_BASELINE_ADJUSTMENT

    # this 5 neuron input is arbitrary, but it seems to work well
    for i in range(stimulus_bump_width): 
        input_bias[int(visual_cue-(stimulus_bump_width-1)+i)] += (motion_coherence/MAX_MOTION_COHERENCE)*k_constant*(1+i)
        input_bias[int(visual_cue+i)] += (motion_coherence/MAX_MOTION_COHERENCE)*k_constant*(stimulus_bump_width-i)

    if DEBUG:
        plt.figure(5500)
        plt.plot(input_bias)
        plt.show()

    return input_bias