"""
Serializes and deserializes images 
after converted to the DataFrame format
"""
import cv2
import numpy as np

# Deserializes image from data frame dump
def deserialize_image(df_dump, width=72, height=48, depth=3):
    # [1:-1] is used to remove '[' and ']' from dataframe string
    df_dump = np.fromstring(df_dump[1:-1], sep=', ', dtype='uint8')
    df_dump = np.resize(df_dump, (height, width, depth))

    return df_dump

# Serializes image to data frame dump
def serialize_image(frame):
    return frame.tolist()