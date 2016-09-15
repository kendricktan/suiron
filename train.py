import json
import numpy as np

from suiron.utils.datasets import get_servo_dataset
from suiron.core.SuironML import get_cnn_model, get_nn_model

# Load image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# Our datasets
x1, servo1 = get_servo_dataset('data/output_2.csv', start_index=35, output=SETTINGS['output'])
x2, servo2 = get_servo_dataset('data/output_3.csv', start_index=5, end_index=520, output=SETTINGS['output'])
X = x1 + x2
SERVO = servo1 + servo2

X = np.array(X)
SERVO = np.array(SERVO) 

# One NN for servo, one for motor
# for now, outputs = 10
servo_model = get_cnn_model('cnn_servo_model', SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'], SETTINGS['output'])
servo_model.fit({'input': X}, {'target': SERVO}, n_epoch=10000,
                validation_set=0.1, show_metric=True, snapshot_epoch=False,
                snapshot_step=500, run_id=SETTINGS['servo_cnn_name'])
servo_model.save(SETTINGS['servo_cnn_name'] + '.ckpt')