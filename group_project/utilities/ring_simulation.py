import matplotlib.pyplot as plt
import numpy as np
from utilities.input_values import DEBUG

def get_perceptual_decision(firing_rate):
    return np.argmax(firing_rate[-1,:])

def ring_simulation(total_time_steps,
                    connectivity_matrix,
                    input_bias,
                    firing_rate,
                    dt=0.005,
                    tau_m=0.1):

    #integrating to future time steps
    for t in range(total_time_steps-1):
        u = connectivity_matrix @ firing_rate[t,:]
        for i in range(len(connectivity_matrix)):
            firing_rate[t+1][i] = firing_rate[t][i] + dt/tau_m * (-firing_rate[t][i] + max(u[i] + input_bias[i],0))
    
    if DEBUG:
        fig = plt.figure(7700)
        ax = fig.add_subplot()
        im = ax.imshow(firing_rate, aspect='auto')
        cbar = fig.colorbar(im)
        plt.show()

    return firing_rate, get_perceptual_decision(firing_rate)