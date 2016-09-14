import cv2
import os
import numpy as np
import time

from suiron.core.SuironIO import SuironIO

suironio = SuironIO()
suironio.init_saving()

print('Warming up...')
#time.sleep(3)

print('Recording data...')
for i in range(60):
    suironio.record_inputs()

print('Saving file...')
suironio.save_inputs()