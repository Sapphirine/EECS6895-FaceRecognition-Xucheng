from PyQt4 import QtCore
import dlib
from caffe_net import *
import glob
import caffe
from caffe.proto import caffe_pb2
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import numpy as np
import cv2

caffe_model = '../deep_model/VGG_FACE.caffemodel'
deploy_file = '../deep_model/VGG_FACE_deploy.prototxt'
mean_file = None
net = Deep_net(caffe_model, deploy_file, mean_file, gpu=True)
db_path = './db'
data_path = '../data/mcem0_sx228'
db = None
label = ['Stranger']
face_info = {}
face_detector = dlib.get_frontal_face_detector()
ldmark_detector = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
ldmarking = True

#  File names
if not os.path.exists(db_path):
    print('Database path is not existed!')
folders = sorted(glob.glob(os.path.join(db_path, '*')))
for name in folders:
    label.append(os.path.basename(name))
    img_list = glob.glob(os.path.join(name, '*.jpg'))

    imgs = [cv2.imread(img) for img in img_list]
    scores, pred_labels, fea = net.classify(imgs, layer_name='fc7')

    fea = np.mean(fea, 0)
    print(fea[:])
    if db is None:
        db = fea.copy()
    else:
        db = np.vstack((db, fea.copy()))

#  Test
if not os.path.exists(data_path):
    print('Data path does not exist!')
img_seq = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
print len(img_seq)
for i in range(len(img_seq)):
    print img_seq[i], i

for img in img_seq:
    frame = cv2.imread(img)
    #   Detect faces
    detected_faces = face_detector(frame, 0)
    if len(detected_faces) > 0:
        print('Number of face detected: {}'.format(len(detected_faces)))

    for k, d in enumerate(detected_faces):
        #   Detection
        landmarks = []
        if ldmarking:
            shape = ldmark_detector(frame, d)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            eye_l = np.mean(landmarks[36:42], axis=0)
            eye_r = np.mean(landmarks[42:48], axis=0)
        crop_face = np.copy(frame[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])
        face_info[k] = ([d.left(), d.top(), d.right(), d.bottom()], landmarks[18:], crop_face)
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 5)
        #   Recognition
        frame_data = [crop_face]
        prob, pred, feature = net.classify(frame_data, layer_name='fc7')
        dist = sklearn.metrics.pairwise.cosine_similarity(feature, db)
        pred = np.argmax(dist, 1)
        dist = np.max(dist, 1)
        threshold = 0.8
        if dist > threshold:
            pred = pred + 1
        else:
            pred = 0
        #   Mark each face with label
        cv2.putText(frame, label[pred], (d.left(), d.bottom() + 30), 0, 1, (0, 255, 0), 3)

    #   Show results
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)  #   25 is better
