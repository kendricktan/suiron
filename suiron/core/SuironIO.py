import time
import random
import numpy as np
import pandas as pd
import cv2, os, serial, csv
import matplotlib.pyplot as plt

from suiron.utils.functions import cnn_to_raw
from suiron.utils.img_serializer import serialize_image
from suiron.utils.file_finder import get_new_filename

class SuironIO:
    """
    Class which handles input output aspect of the suiron 
    - Reads inputs from webcam and normalizes them
    - Also reads serial input and write them to file
    """

    # Constructor
    def __init__(self, width=72, height=48, depth=3, serial_location='/dev/ttyUSB0', baudrate=57600):
        # Image settings
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)

        # Video IO 
        self.cap =  cv2.VideoCapture(0) # Use first capture device

        # Serial IO
        self.ser = None
        if os.path.exists(serial_location):
            print('Found %s, starting to read from it...' % serial_location)
            self.ser = serial.Serial(serial_location, baudrate)        
        self.outfile = None        

        # In-memory variable to record data
        # to prevent too much I/O
        self.frame_results = []
        self.servo_results = []
        self.motorspeed_results = [] 
    
    """ Functions below are used for inputs (recording data) """
    # Initialize settings before saving 
    def init_saving(self, folder='data', filename='output_', extension='.csv'):
        fileoutname = get_new_filename(folder=folder, filename=filename, extension=extension)

        # Filename to save serial data and image data
        # Output file
        outfile = open(fileoutname, 'w') # Truncate file first
        self.outfile = open(fileoutname, 'a')

    # Saves both inputs
    def record_inputs(self):
        # Frame is just a numpy array
        frame = self.get_frame()

        # Serial inputs is a dict with key 'servo', and 'motor'
        serial_inputs = self.get_serial()

        # If its not in manual mode then proceed
        if serial_inputs:
            servo = serial_inputs['servo'] 
            motor = serial_inputs['motor'] 

            # Append to memory
            # tolist so it actually appends the entire thing
            self.frame_results.append(serialize_image(frame))
            self.servo_results.append(servo)
            self.motorspeed_results.append(motor)

    # Get motor inputs, steering inputs etc
    def get_serial(self):
        # For debugging
        serial_raw = '-1,-1\n'
        if self.ser:
            # Polling for consistent data
            self.ser.write('d')
            serial_raw = self.ser.readline()
        serial_processed = self.normalize_serial(serial_raw)
        return serial_processed

    # Gets frame
    def get_frame(self):
        ret, frame = self.cap.read()

        # If we get a frame, save it
        if not ret:
            raise IOError('No image found!')

        frame = self.normalize_frame(frame)
        return frame

    # Gets frame for prediction
    def get_frame_prediction(self):
        ret, frame = self.cap.read()

        # if we get a frame
        if not ret:
            raise IOError('No image found!')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        frame = frame.astype('uint8')

        return frame
    

    # Normalizes inputs so we don't have to worry about weird
    # characters e.g. \r\n
    def normalize_serial(self, line):
        # Assuming that it receives 
        # servo, motor
        
        # 'error' basically means that 
        # its in manual mode
        try:
            line = line.replace('\n', '').split(',')
            line_dict = {'servo': int(line[0]), 'motor': int(line[1])}
            return line_dict
        except:
            return None

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
            'servo': self.servo_results,
            'motor': self.motorspeed_results
        }
        df = pd.DataFrame(raw_data, columns=['image', 'servo', 'motor'])
        df.to_csv(self.outfile)

    """ Functions below are used for ouputs (controlling servo/motor) """    
    # Controls the servo given the numpy array outputted by
    # the neural network
    def servo_write(self, np_y):
        servo_out = cnn_to_raw(np_y)

        if (servo_out < 90):
            servo_out *= 0.85

        elif (servo_out > 90):
            servo_out *= 1.15

        self.ser.write('steer,' + str(servo_out) + '\n') 
        time.sleep(0.02)

    # Sets the motor at a fixed speed
    def motor_write_fixed(self):    
        self.ser.write('motor,80\n')
        time.sleep(0.02)

    # Stops motors
    def motor_stop(self):      
        self.ser.write('motor,90\n')
        time.sleep(0.02)

    # Staightens servos
    def servo_straighten(self):
        self.ser.write('steer,90')
        time.sleep(0.02)
        
    def __del__(self):
        if self.outfile:
            self.outfile.close()