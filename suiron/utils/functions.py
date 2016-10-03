import numpy as np

#Map function from arduino
def arduino_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Converts servo values to target (Y) values
# for the neural network
def servo_to_target(y, output=10):
    # Servo values
    # Map from 40-140 to 1-10 and
    # Convert to angle and then convert to radians
    degrees = arduino_map(y, 40.0, 150.0, 0.0, 90.0)
    radians = degrees * np.pi / 180.0
    return [radians] 

# Converts neural network outputs 
# to servo values
def target_to_servo(y, output=10):
    # Get highest index value and map
    # it back
    radians = y[np.argmax(y)]

    # output is in radians
    # convert to degrees
    degrees = radians * 180.0 / np.pi

    print(degrees)

    # degrees to output
    y_ = arduino_map(degrees, 0.0, 90.0, 40.0, 150.0)

    return y_