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
db = None
label = ['Stranger']

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
print label

#   Read images
test_person = cv2.imread('../data/Al_Pacino.jpg')
test_data = [test_person]
prob, pred, feature = net.classify(test_data, layer_name='fc7')
dist = sklearn.metrics.pairwise.cosine_similarity(feature, db)
pred = np.argmax(dist, 1)
dist = np.max(dist, 1)
threshold = 0.6
if dist < threshold:
    pred = pred + 1
else:
    pred = 0

#   Show result
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.putText(test_person, label[pred], (0, 120), 0, 1, (0, 0, 255), 2)
cv2.imshow("Result", test_person)
cv2.waitKey(0)