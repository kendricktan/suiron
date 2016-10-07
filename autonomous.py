import time
import json
import numpy as np

from suiron.core.SuironIO import SuironIO
from suiron.core.SuironML import get_cnn_model
from suiron.utils.functions import cnn_to_raw

# Image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# IO Class to serial ports (read and write em)
print('Initiating I/O operations...')
suironio = SuironIO(width=SETTINGS['width'], height=SETTINGS['height'], depth=SETTINGS['depth'])
suironio.init_saving()
suironio.motor_stop()

# CNN Model
print('Initiating CNN model...')
servo_model = get_cnn_model(SETTINGS['servo_cnn_name'], SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'])
servo_model.load(SETTINGS['servo_cnn_name'] + '.ckpt')

print('Warming up camera...')
time.sleep(5)
# Auto calibration for camera
for i in range(50):
    suironio.get_frame()

raw_input('Press any key to autonomous mode now...')
suironio.motor_write_fixed()
while True:
    try:
        # Get current frame
        c_frame = suironio.get_frame_prediction()

        # Get predictions
        y_ = servo_model.predict([c_frame])        
        s_o = cnn_to_raw(y_[0])

        # Write outputs to servo
        suironio.servo_write(y_[0])

    except KeyboardInterrupt:
        suironio.motor_stop()
        suironio.servo_straighten()

print('Exiting autonomous mode...')
