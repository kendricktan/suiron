import os

# Check if folder exists
def check_folder_exists(folder='data'):
    if not os.path.exists(folder):
        os.mkdir(folder)

# Get latest filename for outfile
def get_new_filename(folder='data', filename='output_', extension='.csv'):
    check_folder_exists(folder)

    iter_name = 0
    while os.path.exists(os.path.join(folder, filename+str(iter_name)+extension)):
        iter_name += 1
    
    fileoutname = filename + str(iter_name) + extension
    fileoutname = os.path.join(folder, fileoutname)

    return fileoutname

# Get latest output filename
def get_latest_filename(folder='data', filename='output_', extension='.csv'):
    check_folder_exists(folder)

    iter_name = 0
    while os.path.exists(os.path.join(folder, filename+str(iter_name)+extension)):
        iter_name += 1

    iter_name -= 1

    if iter_name == -1:
        raise IOError('No file found!')
    
    fileoutname = filename + str(iter_name) + extension
    fileoutname = os.path.join(folder, fileoutname)

    return fileoutname