import cv2, os, serial

class SurionCV:
    """
    Class which handles the vision aspect of the suiron 
    - Reads inputs from webcam and normalizes them
    - Also reads serial input and write them to file
    """

    # Constructor
    def __init__(self, width=1080, height=720, baudrate=9600):
        # Image settings
        self.width = width
        self.height = height

        # Video IO 
        self.cap =  cv2.VideoCapture(0) # Use first capture device
        self.vid_out = None

        # Serial IO
        #self.ser = serial.Serial('/dev/tty.usbserial', baudrate)
        self.inserial = None
    
    # Initialize settings before saving 
    def init_saving(self, folder='data', filename='output_'):
        # Folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Filename for video
        iter_name = 0
        while os.path.exists(os.path.join(folder, filename+str(iter_name)+'.avi')):
            iter_name += 1
        fileoutname = filename + str(iter_name) + '.avi'
        fileoutname = os.path.join(folder, fileoutname)

        # Encoders for video
        fourcc = cv2.cv.FOURCC(*'XVID')
        self.vid_out = cv2.VideoWriter(fileoutname, fourcc, 30, (self.width, self.height))

        # Filename to save serial data
        inserial_filename = os.path.join(folder, filename+str(iter_name)+'.txt')
        self.inserial = open(inserial_filename, 'w') # Truncate file first
        self.inserial = open(inserial_filename, 'a') # Then append to it

    # Saves both inputs
    def save_all(self):
        self.save_serial()
        self.save_frame()

    # Save motor inputs, steering inputs etc
    def save_serial(self):
        #serial_raw = self.ser.readline()
        #serial_processed = self.normalize_serial(serial_raw)
        self.inserial.write('1' + '\n')

    # Saves frame
    def save_frame(self):
        if self.vid_out is None:
            raise EnvironmentError('Please run init_encoding() before running save_frame()!')
            
        ret, frame = self.cap.read()

        # If we get a frame, save it
        if ret:
            frame = self.normalize_frame(frame)
            # Saves frame
            self.vid_out.write(frame)

    # Normalizes inputs so we don't have to worry about weird
    # characters e.g. \r\n
    @staticmethod
    def normalize_serial(line):
        return line

    # Normalizes frame so we don't have BGR as opposed to RGB
    @staticmethod
    def normalize_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    # Destructor
    def __del__(self):
        # Close video
        if self.vid_out:
            self.vid_out.release()

        # Close serial
        if self.inserial:
            self.inserial.close()
