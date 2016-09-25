import numpy as np

#Map function from arduino
def arduino_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

# Converts servo values to target (Y) values
# for the neural network
def servo_to_target(y, output=10):
    # Servo values
    # Map from 40-140 to 1-10 and
    # make them into a 1x10 dimensional array
    # where array[N] is 1, and the rest is 0
    # output-1 because computers count from 0
    y_ = np.zeros(output)
    index_ = arduino_map(y, 40, 140, 0, output-1)
    y_[index_] = 1
    return y_

# Converts neural network outputs 
# to servo values
def target_to_servo(y, output=10):
    # Get highest index value and map
    # it back
    y_ = np.argmax(y)
    y_ = arduino_map(y_, 0, output-1, 40, 150)
    return y_