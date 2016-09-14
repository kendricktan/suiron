import cv2
import os
import numpy as np

from cv.SurionCV import SurionCV

surioncv = SurionCV()
surioncv.init_saving()

for i in range(60):
    surioncv.save_all()