import numpy as np
import cv2
import pandas as pd

WIDTH = 72 
HEIGHT = 48
DEPTH = 3

def visualize_data(filename):
    data = pd.DataFrame.from_csv(filename)

    for i in data.index:
        cur_img = data['image'][i]
        cur_throttle = int(data['throttle'][i])
        
        # [1:-1] is used to remove '[' and ']' from string 
        cur_img_array = np.fromstring(cur_img[1:-1], sep=',', dtype='uint8')
        cur_img_array = np.resize(cur_img_array, (HEIGHT, WIDTH, DEPTH))

        cur_img_array = cv2.resize(cur_img_array, (480, 320), interpolation=cv2.INTER_CUBIC)

        # Extra info
        cv2.putText(cur_img_array, "frame: {}".format(str(i)), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.line(cur_img_array, (240, 300), (240+cur_throttle, 200), (0, 255, 0), 3)

        # Show frame
        # Convert to BGR cause thats how OpenCV likes it
        cv2.imshow('frame', cv2.cvtColor(cur_img_array, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
