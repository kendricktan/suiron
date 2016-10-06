import sys, os
import json
import numpy as np

from suiron.utils.datasets import get_servo_dataset
from suiron.core.SuironML import get_cnn_model 

# Load image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# Our datasets
print('[!] Loading dataset...')
X = []
SERVO = []
DATA_FILES = ['data/output_0.csv', 'data/output_1.csv', 'data/output_2.csv', 'data/output_3.csv', 'data/output_4.csv']
for d in DATA_FILES:
    c_x, c_servo = get_servo_dataset(d)
    X = X + c_x
    SERVO = SERVO + c_servo

X = np.array(X)
SERVO = np.array(SERVO) 
print('[!] Finished loading dataset...')

# One NN for servo, one for motor
# for now, outputs = 10
servo_model = get_cnn_model(SETTINGS['servo_cnn_name'], SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'])

# Loads previous model if specified
if len(sys.argv) > 1:
    servo_model.load(sys.argv[1])

servo_model.fit({'input': X}, {'target': SERVO}, n_epoch=10000,
                validation_set=0.1, show_metric=True, snapshot_epoch=False,
                snapshot_step=10000, run_id=SETTINGS['servo_cnn_name'])
servo_model.save(SETTINGS['servo_cnn_name'] + '.ckpt')