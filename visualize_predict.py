import sys
import json
import numpy as np

import tensorflow as tf

from suiron.utils.file_finder import get_latest_filename
from suiron.core.SuironML import get_cnn_model, get_nn_model
from suiron.core.SuironVZ import visualize_data

# Image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)


# Load up our CNN
servo_model = None
servo_model = get_cnn_model(SETTINGS['servo_cnn_name'], SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'])
try:        
    servo_model.load(SETTINGS['servo_cnn_name'] + '.ckpt')
except Exception as e:
    print(e)
    
# Load up our NN
# For now NN determines speed '79' is the best speed?
motor_model = None
# motor_model = get_nn_model(SETTINGS['motor_nn_name'])
# try:        
#     motor_model.load(SETTINGS['motor_nn_name'] + '.ckpt')
# except Exception as e:
#     print(e)

# Visualize latest filename
filename = get_latest_filename() 

# If we specified which file
if len(sys.argv) > 1:
    filename = sys.argv[1]

# Visualize it
visualize_data(filename, SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'], cnn_model=servo_model, nn_model=motor_model)