import sys, os
import json
import numpy as np

from suiron.utils.datasets import get_motor_dataset
from suiron.core.SuironML import get_cnn_model, get_nn_model

# Load image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# Our datasets
print('[!] Loading dataset...')
SERVO = []
MOTOR = []
DATA_FILES = ['data/output_0.csv', 'data/output_1.csv', 'data/output_2.csv', 'data/output_3.csv', 'data/output_4.csv']
for d in DATA_FILES:
    c_servo, c_motor = get_motor_dataset(d)
    MOTOR = MOTOR + c_motor
    SERVO = SERVO + c_servo

MOTOR = np.array(MOTOR)
SERVO = np.array(SERVO) 
print('[!] Finished loading dataset...')

# One NN for servo, one for motor
# for now, outputs = 10
motor_model = get_nn_model(SETTINGS['motor_nn_name'])

# Loads previous model if specified
if len(sys.argv) > 1:
    motor_model.load(sys.argv[1])

motor_model.fit({'input': SERVO}, {'target': MOTOR}, n_epoch=1000,
                validation_set=0.1, show_metric=True, snapshot_epoch=False,
                snapshot_step=10000, run_id=SETTINGS['motor_nn_name'])
motor_model.save(SETTINGS['motor_nn_name'] + '.ckpt')