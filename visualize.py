from suiron.core.SuironVZ import visualize_data
from suiron.utils.file_finder import get_latest_filename

# Visualize latest filename
filename = get_latest_filename() 
visualize_data(filename)