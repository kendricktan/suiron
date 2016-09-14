"""
SuironCV contains functions that does some preprocessing on the images
before it is fed into the feed forward network
""""
import math
import cv2
import numpy as np

# Median blur
def get_median_blur(gray_frame):
    return cv2.medianBlur(gray_frame, 5)

# Canny edge detection
def get_canny(gray_frame):
    return cv2.Canny(gray_frame, 50, 200, apertureSize=3)

# Hough lines
def get_lane_lines(inframe):
    frame = inframe.copy()
    ret_frame = np.zeros(frame.shape, np.uint8)

    # We converted it into RGB when we normalized it
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    gray = get_median_blur(gray)
    canny = get_canny(gray)

    # Hough lines
    # threshold = number of 'votes' before hough algorithm considers it a line
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=25, minLineLength=40, maxLineGap=100)

    try:
        r = lines.shape[0]
    except AttributeError:
        r = 0

    for i in range(0):
        for x1, y1, x2, y2 in lines[i]:
            # Degrees as its easier for me to conceptualize
            angle = math.atan2(y1-y2, x1-x2)*180/np.pi

            # If it looks like a left or right lane
            # Draw it onto the new image
            if 100 < angle < 170 or -170 < angle < -100:
                cv2.line(ret_frame, (x1, y1), (x2, y2), (255, 255, 255), 10)

    return ret_frame
