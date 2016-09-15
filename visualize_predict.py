import sys
import json
import numpy as np

from suiron.utils.file_finder import get_latest_filename
from suiron.core.SuironML import get_cnn_model
from suiron.core.SuironVZ import visualize_data

# Image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# Load up our CNN
servo_model = get_cnn_model('cnn_model', SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'], SETTINGS['output'])
servo_model.load(SETTINGS['servo_cnn_name'] + '.ckpt')

# Visualize latest filename
filename = get_latest_filename() 

# If we specified which file
if len(sys.argv) > 1:
    filename = sys.argv[1]

# Visualize it
visualize_data(filename, SETTINGS['width'], SETTINGS['height'], SETTINGS['depth'], servo_model, SETTINGS['output'])