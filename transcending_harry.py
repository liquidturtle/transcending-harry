% transcending harry facial landmarks and body pose recognition
% Jan 2022
% Tobias Ulrich

import cv2
import time

cap = cv2.VideoCapture("~/home/Tobias/Documents/dynamicframe")

while True:
	success, img = cap.read()
	cv2.imshow("Image";img)
	cv2.waitKey(1)
