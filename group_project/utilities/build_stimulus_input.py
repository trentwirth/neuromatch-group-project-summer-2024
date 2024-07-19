import numpy as np
import matplotlib.pyplot as plt
from utilities.input_values import CONSTANT_INPUT_BIAS, K_CONSTANT, RANDOM_BASELINE_ADJUSTMENT

def build_stimulus_input(firing_rate: np.ndarray,
                         init_input_bias: np.ndarray,
                         visual_cue: int = 225):

    

    firing_rate[0,:]=firing_rate[-1,:]
    input_bias = np.random.random(size=len(init_input_bias)) - RANDOM_BASELINE_ADJUSTMENT

    for i in range(5):
        input_bias[visual_cue-4+i] += K_CONSTANT*(1+i)*CONSTANT_INPUT_BIAS
        input_bias[visual_cue+i] += K_CONSTANT*(5-i)*CONSTANT_INPUT_BIAS

    plt.figure(5500)
    plt.plot(input_bias)
    plt.show()

    return input_bias