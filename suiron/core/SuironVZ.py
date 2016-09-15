import numpy as np
import cv2
import pandas as pd

from suiron.utils.functions import target_to_servo
from suiron.utils.img_serializer import deserialize_image

# Visualize images
# With and without any predictions
def visualize_data(filename, width=72, height=48, depth=3, model=None, output=10):
    """
    When model is specified it'll show what the model predicts (red)
    as opposed to what inputs it actually received (green)
    """
    data = pd.DataFrame.from_csv(filename)

    for i in data.index:
        cur_img = data['image'][i]
        cur_throttle = int(data['servo'][i])
        cur_motor = int(data['motor'][i])
        
        # [1:-1] is used to remove '[' and ']' from string 
        cur_img_array = deserialize_image(cur_img)        
        y_input = cur_img_array.copy() # NN input

        # And then rescale it so we can more easily preview it
        cur_img_array = cv2.resize(cur_img_array, (480, 320), interpolation=cv2.INTER_CUBIC)

        # Extra debugging info (e.g. steering etc)
        cv2.putText(cur_img_array, "frame: %s" % str(i), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.line(cur_img_array, (240, 300), (240-(90-cur_throttle), 200), (0, 255, 0), 3)

        # If we wanna visualize our model
        if model:
            y = model.predict([y_input])
            servo_out = target_to_servo(y[0], output)
            cv2.line(cur_img_array, (240, 300), (240-(90-servo_out), 200), (0, 0, 255), 3)

        # Show frame
        # Convert to BGR cause thats how OpenCV likes it
        cv2.imshow('frame', cv2.cvtColor(cur_img_array, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break