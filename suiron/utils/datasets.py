"""
datasets.py provides functions to help condense data 'collect.py' into
numpy arrays which can be fed into the CNN/NN
"""

import numpy as np
import pandas as pd

from suiron.utils.img_serializer import deserialize_image
from suiron.utils.functions import arduino_map

def get_servo_dataset(filename, start_index=0, end_index=None, output=10):
    data = pd.DataFrame.from_csv(filename)

    # Outputs
    x = []

    # Servo ranges from 40-140
    # Gonna map them from 1-10
    servo = []

    for i in data.index[start_index:end_index]:
        # Don't want noisy data
        if data['servo'][i] < 40 or data['servo'][i] > 140:
            continue

        x.append(deserialize_image(data['image'][i]))

        # Servo values
        # Map from 40-140 to 1-10 and
        # make them into a 1x10 dimensional array
        # where array[N] is 1, and the rest is 0
        # output-1 because computers count from 0
        y_ = np.zeros(output)
        index_ = arduino_map(data['servo'][i], 40, 140, 0, output-1)
        y_[index_] = 1
        servo.append(y_)

    return x, servo