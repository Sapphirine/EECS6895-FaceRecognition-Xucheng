from PyQt4 import QtCore
import dlib
import numpy as np
import cv2

img = cv2.imread('../data/Al_Pacino.jpg')

face_detector = dlib.get_frontal_face_detector()
ldmark_detector = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
det = face_detector(img, 0)
for k, d in enumerate(det):
    landmarks1 = []
    shape = ldmark_detector(img, d)
    landmarks1 = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    eye_l = np.mean(landmarks1[36:42], axis=0)
    eye_r = np.mean(landmarks1[42:48], axis=0)
crop_face = np.copy(img[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])
cv2.imwrite('../data/Al_Pacino.jpg', crop_face)
