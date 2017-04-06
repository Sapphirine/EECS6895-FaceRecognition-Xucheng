from PyQt4 import QtCore
import dlib
from caffe_net import *
import glob
import caffe
from caffe.proto import caffe_pb2
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

caffe_model = '../deep_model/VGG_FACE.caffemodel'
deploy_file = '../deep_model/VGG_FACE_deploy.prototxt'
mean_file = None
net = Deep_net(caffe_model, deploy_file, mean_file, gpu=True)
db_path = './db'
db = None
label = ['Stranger']

transformer = net.get_transformer(deploy_file, mean_file)

#  file names
if not os.path.exists(db_path):
    print('Database path is not existed!')
folders = sorted(glob.glob(os.path.join(db_path, '*')))

# for name in folders:
#     print('loading {}:'.format(name))
#     label.append(os.path.basename(name))
#     img_list = glob.glob(os.path.join(name, '*.jpg'))
#     imgs = [cv2.imread(img) for img in img_list]
#
#     scores, pred_labels, fea = net.classify(imgs, layer_name='fc7')
#     print scores
#     print pred_labels
#     print fea
#
#     fea = np.mean(fea, 0)
#     print(fea[:])
#     if db is None:
#         db = fea.copy()
#     else:
#         db = np.vstack((db, fea.copy()))

#   Read images
img1 = cv2.imread("./db/person3/001.jpg")
img2 = cv2.imread("./db/person3/002.jpg")

#   Crop faces
face_detector = dlib.get_frontal_face_detector()
ldmark_detector = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
det1 = face_detector(img1, 0)
for k, d in enumerate(det1):
    landmarks1 = []
    shape1 = ldmark_detector(img1, d)
    landmarks1 = [(shape1.part(i).x, shape1.part(i).y) for i in range(68)]
    eye_l = np.mean(landmarks1[36:42], axis=0)
    eye_r = np.mean(landmarks1[42:48], axis=0)
crop_face1 = np.copy(img1[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])

det2 = face_detector(img2, 0)
for k, d in enumerate(det2):
    landmarks2 = []
    shape2 = ldmark_detector(img2, d)
    landmarks2 = [(shape2.part(i).x, shape2.part(i).y) for i in range(68)]
    eye_l = np.mean(landmarks2[36:42], axis=0)
    eye_r = np.mean(landmarks2[42:48], axis=0)
crop_face2 = np.copy(img2[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])

#   Write images
cv2.imwrite('./db/person3/001.jpg', crop_face1)
cv2.imwrite('./db/person3/002.jpg', crop_face2)

#   Read images
person1 = [crop_face1]
person2 = [crop_face2]

#   Generate features
scores, pred_labels, fea = net.classify(person1, layer_name='fc7')
scores2, pred_labels2, fea2 = net.classify(person2, layer_name='fc7')
dist = sklearn.metrics.pairwise.cosine_similarity(fea, fea2)
pred = np.argmax(dist, 1)
dist = np.max(dist, 1)

#   Show comparison
plt.figure(num='Person', figsize=(8,8))
plt.subplot(1, 2, 1)
plt.title('Person 1')
plt.text(60, 100, dist, fontsize = 12)
img1 = mpimg.imread('./db/person3/001.jpg')
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.title('Person 2')
img2 = mpimg.imread('./db/person3/002.jpg')
plt.imshow(img2)
plt.show()