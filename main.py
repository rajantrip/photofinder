import face_recognition
import imutils
from imutils import paths
import pickle
import time
import cv2
import os
import glob

ext = ['png', 'jpg', 'gif']
path = (r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images')
#imagePaths = []
#[imagePaths.extend(glob.glob(path + '*.' + e)) for e in ext]
#imagePaths = [f for f in glob.glob(path+'*.jpg')]
imagePaths = list(paths.list_images(path))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    nameraw = imagePath.split("\\")[-1]
    name = nameraw.split(".")[0]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
data = {"encodings": knownEncodings, "names": knownNames}
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()

