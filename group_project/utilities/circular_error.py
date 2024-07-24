import numpy as np

def circular_error(estimate_angle, percept_decision):
    error = np.abs((estimate_angle - percept_decision + 180) % 360 - 180)
    return error