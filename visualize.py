import json

from suiron.core.SuironVZ import visualize_data
from suiron.utils.file_finder import get_latest_filename

# Load image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# Visualize latest filename
filename = get_latest_filename() 
visualize_data(filename, width=SETTINGS['width'], height=SETTINGS['height'], depth=SETTINGS['depth'])