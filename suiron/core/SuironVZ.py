import cv2
import numpy as np
import pandas as pd

WIDTH = 75
HEIGHT = 50
DEPTH = 3

def visualize_data(filename):
    data = pd.DataFrame.from_csv(filename)

    for i in data.index:
        cur_img = data['image'][i]
        cur_throttle = data['throttle'][i]
        
        # [1:-1] is used to remove '[' and ']' from string 
        cur_img_array = np.fromstring(cur_img[1:-1], sep=',', dtype='uint8')
        cur_img_array = np.resize(cur_img_array, (HEIGHT, WIDTH, DEPTH))

        cur_img_array = cv2.resize(cur_img_array, (480, 320), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('frame', cv2.cvtColor(cur_img_array, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

