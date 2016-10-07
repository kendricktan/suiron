import numpy as np
import cv2
import pandas as pd

from suiron.utils.functions import raw_to_cnn, cnn_to_raw, raw_motor_to_rgb
from suiron.utils.img_serializer import deserialize_image

# Visualize images
# With and without any predictions
def visualize_data(filename, width=72, height=48, depth=3, cnn_model=None):
    """
    When cnn_model is specified it'll show what the cnn_model predicts (red)
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

        # Motor values
        # RGB
        cv2.line(cur_img_array, (50, 160), (50, 160-(90-cur_motor)), raw_motor_to_rgb(cur_motor), 3)

        # If we wanna visualize our cnn_model
        if cnn_model:
            y = cnn_model.predict([y_input])
            servo_out = cnn_to_raw(y[0])         
            cv2.line(cur_img_array, (240, 300), (240-(90-int(servo_out)), 200), (0, 0, 255), 3)

            # Can determine the motor our with a simple exponential equation
            # x = abs(servo_out-90)
            # motor_out = (7.64*e^(-0.096*x)) - 1
            # motor_out = 90 - motor_out
            x_ = abs(servo_out - 90)
            motor_out = (7.64*np.e**(-0.096*x_)) - 1
            motor_out = int(80 - motor_out) # Only wanna go forwards
            cv2.line(cur_img_array, (100, 160), (100, 160-(90-motor_out)), raw_motor_to_rgb(motor_out), 3)
            print(motor_out, cur_motor)

        # Show frame
        # Convert to BGR cause thats how OpenCV likes it
        cv2.imshow('frame', cv2.cvtColor(cur_img_array, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break