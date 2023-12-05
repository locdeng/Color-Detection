import cv2
from util import get_limits
from color_api import color_histogram_test_image
from color_api import color_histogram_training_image
from color_api import knn
from PIL import Image
import numpy as np
import os.path
import os


# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print('training data is ready, classifier is loading...')


red = [0, 0, 255]
orange = [0, 165, 255]
yellow = [0, 255, 255]
green = [0, 255, 0]
blue = [255, 0, 0]
violet = [138, 43, 226]

color_bgr_values = dict(red=(0, 0, 255), orange=(0, 165, 255), yellow=(0, 255, 255), green=(0, 255, 0),
                        blue=(255, 0, 0), violet=(138, 43, 226))

color_ranges = {
    'red': [get_limits(red)],
    'orange': [get_limits(orange)],
    'yellow': [get_limits(yellow)],
    'green': [get_limits(green)],
    'blue': [get_limits(blue)],
    'violet': [get_limits(violet)]
}

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, color_limits in color_ranges.items():
        for lowerLimit, upperLimit in color_limits:
            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

            # Apply erosion to remove noise
            kernel = np.ones((5, 5), np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            opened_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel)
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

            # Apply dilation to restore the original size of training_dataset regions
            dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

            mask_ = Image.fromarray(dilated_mask)

            bbox = mask_.getbbox()

            if bbox is not None:
                x1, y1, x2, y2 = bbox

                # Get the training_dataset name and use it to retrieve the BGR value
                rectangle_color = color_bgr_values[color_name]

                # Draw a rectangle with the detected training_dataset
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rectangle_color, 5)

                # Add the training_dataset name near the bounding box
                cv2.putText(frame, color_name, (x1, y1 - 10), font, 0.9, rectangle_color, 2, cv2.LINE_AA)

    cv2.imshow('color_detector', frame)

    color_histogram_test_image.color_histogram_of_test_image(frame)
    color_name = knn.main('training.data', 'test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
