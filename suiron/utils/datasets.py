"""
datasets.py provides functions to help condense data 'collect.py' into
numpy arrays which can be fed into the CNN/NN
"""

import numpy as np
import pandas as pd

from suiron.utils.img_serializer import deserialize_image
from suiron.utils.functions import servo_to_target

def get_servo_dataset(filename, start_index=0, end_index=None, output=10):
    data = pd.DataFrame.from_csv(filename)

    # Outputs
    x = []

    # Servo ranges from 40-150
    # Gonna map them from 1-10
    servo = []

    for i in data.index[start_index:end_index]:
        # Don't want noisy data
        if data['servo'][i] < 40 or data['servo'][i] > 150:
            continue

        # Append
        x.append(deserialize_image(data['image'][i]))
        servo.append(servo_to_target(data['servo'][i], output))

    return x, servo