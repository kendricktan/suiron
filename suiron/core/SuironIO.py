import numpy as np
import pandas as pd
import cv2, os, serial, csv
import matplotlib.pyplot as plt

class SuironIO:
    """
    Class which handles input output aspect of the suiron 
    - Reads inputs from webcam and normalizes them
    - Also reads serial input and write them to file
    """

    # Constructor
    def __init__(self, width=75, height=50, serial_location='/dev/tty.usbserial', baudrate=9600):
        # Image settings
        self.width = width
        self.height = height

        # Video IO 
        self.cap =  cv2.VideoCapture(0) # Use first capture device

        # Serial IO
        #self.ser = serial.Serial(serial_location, baudrate)
        self.outfile = None

        # In-memory variable to record data
        # to prevent too much I/O
        self.frame_results = []
        self.throttle_results = []
        self.motorspeed_results = [] 
    
    # Initialize settings before saving 
    def init_saving(self, folder='data', filename='output_'):
        # Folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Filename for video
        iter_name = 0
        while os.path.exists(os.path.join(folder, filename+str(iter_name)+'.csv')):
            iter_name += 1
        fileoutname = filename + str(iter_name) + '.csv'
        fileoutname = os.path.join(folder, fileoutname)

        # Filename to save serial data and image data
        # Output file
        outfile = open(fileoutname, 'w') # Truncate file first
        self.outfile = open(fileoutname, 'a')

    # Saves both inputs
    def record_inputs(self):
        frame = self.get_frame()
        inserial = 1#self.get_serial()

        # Append to memory
        self.frame_results.append(frame.tolist())
        self.throttle_results.append(inserial)

    # Get motor inputs, steering inputs etc
    def get_serial(self):
        serial_raw = self.ser.readline()
        serial_processed = self.normalize_serial(serial_raw)

        return serial_processed

    # Gets frame
    def get_frame(self):
        ret, frame = self.cap.read()

        # If we get a frame, save it
        if not ret:
            return None

        frame = self.normalize_frame(frame)
        return frame

    # Normalizes inputs so we don't have to worry about weird
    # characters e.g. \r\n
    def normalize_serial(self, line):
        return line

    # Normalizes frame so we don't have BGR as opposed to RGB
    def normalize_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        frame = frame.flatten()
        frame = frame.astype('uint8')
        return frame

    # Saves files
    def save_inputs(self):
        raw_data = {
            'image': self.frame_results, 
            'throttle': self.throttle_results
        }
        df = pd.DataFrame(raw_data, columns=['image', 'throttle'])
        df.to_csv(self.outfile)

    def __del__(self):
        if self.outfile:
            self.outfile.close()