import numpy as np

def build_connectivity_matrix(theta_neurons: np.ndarray = None,
                              max_neuron_number: int = 50,
                              W_0 = -2,
                              W_1 = 1):
#The W_{ij} connectivity in the firing rate equation. Initialised randomly.
#Remember to keep pre-synapse and post synapse order in W, the connectivity matrix
    connectivity_matrix = np.random.random(size=(max_neuron_number, max_neuron_number)) 
    
    if theta_neurons is None:
        raise ValueError("No theta neurons provided.")

    for i in range(max_neuron_number):
        for j in range(max_neuron_number):
            connectivity_matrix[i][j] = W_1 * np.cos(theta_neurons[i]-theta_neurons[j]) + W_0 

    return connectivity_matrix