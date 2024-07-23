# Retrieve the data
import pandas as pd
import numpy as np
import requests
from utilities.input_values import EXPERIMENTAL_DATA_PATH

def get_data():

    df = pd.read_csv(EXPERIMENTAL_DATA_PATH)

    # Convert cartesian coordinates to radians, then convert to degrees
    df['estimate_angle'] = np.degrees(np.arctan2(df['estimate_y'], df['estimate_x']))

    # Normalize the angle to be in the range of [0, 360)
    df['estimate_angle'] = df['estimate_angle'] % 360

    # Calculate the absolute angular error between the participant estimate,
    #           and the motion direction of the stimulus
    df['absolute_angular_error_stimulus'] = np.abs((df['estimate_angle'] - df['motion_direction'] + 180) % 360 - 180)

    # Calculate the absolute angular error between the participant estimate,
    #           and the MEAN motion direction ("prior_mean" = 225)
    df['absolute_angular_error_prior_mean'] = np.abs((df['estimate_angle'] - 225 + 180) % 360 - 180)

    df = df.dropna(subset=['estimate_angle'])

    return df