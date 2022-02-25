import face_recognition
import imutils
from imutils import paths
import pickle
import time
import cv2
import os
from flask import Flask, request, jsonify
from PIL import Image
import os.path
import json
import pandas as pd
import numpy as np
import uuid

app = Flask(__name__)

test = os.path.basename('.')

app.config['test'] = test


@app.route('/api/gemsattendance', methods=['GET', 'POST'])
def upload_file():
    #output_dir = r'C:\Users\Abhi\Desktop\App'
    #path = output_dir + str(count) + '.jpg'
    path1 = (r'C:\Users\Abhi\PycharmProjects\gemsimg\cropped')
    file = request.files['image']
    f = os.path.join(app.config['test'], file.filename)
    file.save(f)
    image = cv2.imread(test + "/" + file.filename)
    os.remove(test + "/" + file.filename)
    filename = "{}.jpg".format(os.getpid())
    cv2.imwrite(filename, image)
    #os.remove(filename)

    data = pickle.loads(open(r'C:\Users\Abhi\PycharmProjects\gemsimg\face_enc', "rb").read())
    faceCascade = cv2.CascadeClassifier(
        r'C:\Users\Abhi\PycharmProjects\gemsimg\haarcascade_frontalface_default.xml')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image,
                                         scaleFactor=1.05,
                                         minNeighbors=6,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces = faceCascade.detectMultiScale(image)

    encodings = face_recognition.face_encodings(rgb)
    names = []

    def combine_horizontally(image_names, padding=20):
        images = []
        max_height = 0  # find the max height of all the images
        total_width = 0  # the total width of the images (horizontal stacking)
        for img in image_names:
            # open all images and find their sizes
            images.append(img)
            image_height = img.shape[0]
            image_width = img.shape[1]
            if image_height > max_height:
                max_height = image_height
            # add all the images widths
            total_width += image_width
        # create a new array with a size large enough to contain all the images
        # also add padding size for all the images except the last one
        final_image = np.zeros((max_height, (len(image_names) - 1) * padding + total_width, 3), dtype=np.uint8)
        current_x = 0  # keep track of where your current image was last placed in the x coordinate
        for image in images:
            # add an image to the final array and increment the x coordinate
            height = image.shape[0]
            width = image.shape[1]
            final_image[:height, current_x:width + current_x, :] = image
            # add the padding between the images
            current_x += width + padding
        return final_image

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

            for ((x, y, w, h), name) in zip(faces, names):
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)

                img = rgb[y:y + h, x:x + w]
                path_file1 = (r'C:\Users\Abhi\PycharmProjects\gemsimg\cropped\%s.jpg' % name)
                cv2.imwrite(path_file1, img)

                if os.path.exists(r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images\{roll}.png'.format(roll=name)):
                    original_img = cv2.imread(r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images\{roll}.png'.format(roll=name))
                elif os.path.exists(r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images\{roll}.jpg'.format(roll=name)):
                    original_img = cv2.imread(r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images\{roll}.jpg'.format(roll=name))
                elif os.path.exists(r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images\{roll}.jpeg'.format(roll=name)):
                    original_img = cv2.imread(r'C:\Users\Abhi\PycharmProjects\gemsimg\Train_images\{roll}.jpeg'.format(roll=name))
                path_file = r'C:\Users\Abhi\PycharmProjects\gemsimg\cropped\{filename}.jpg'.format(x=path1.split('/')[-1].split('.')[0], filename=name)
                imgHor = combine_horizontally([img, original_img])
                cv2.imwrite(path_file, imgHor)

                print(len(set(names)))
    return jsonify(names)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug='TRUE', port=8080)


