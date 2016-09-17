import json
import numpy as np

from suiron.core.SuironIO import SuironIO
from suiron.core.SuironML import get_cnn_model

# Image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# IO Class to serial ports (read and write em)
print('Initiating I/O operations...')
suironio = SuironIO(width=SETTINGS['width'], height=SETTINGS['height'], depth=SETTINGS['depth'])
suironio.init_saving()
suironio.init_writing(output=SETTINGS['output'])

# CNN Model
print('Initiating CNN model...')
servo_model = get_cnn_model('cnn_model', SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'], SETTINGS['output'])
servo_model.load(SETTINGS['servo_cnn_name'] + '.ckpt')

print('Warming up camera...')
time.sleep(5)
# Auto calibration for camera
for i in range(50):
    suironio.get_frame()

print('Entering autonomous mode now...')
while True:
    try:
        # Get current frame
        c_frame = suironio.get_frame()

        # Get predictions
        y_ = servo_model.predict([c_frame])

        # Write outputs to servo
        suironio.servo_write(y_[0])

    except KeyboardInterrupt:
        break

print('Exiting autonomous mode...')
