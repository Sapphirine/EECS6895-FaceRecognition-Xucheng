import numpy as np
import cv2

#   Unsolve issue
cap = cv2.VideoCapture("00-1.avi")

while(1):
    res, frame = cap.read()
    print res
    if res == True:
        cv2.imshow('frame', frame)
    else:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

