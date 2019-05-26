import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIRS = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIRS, 'images')

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_trains = []
y_labels = []

for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            print(label_ids)

            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)

            for(x, y, w, h) in faces:
                roi = image_array[x:x+w, y:y+h]
                x_trains.append(roi)
                y_labels.append(id_)

print(y_labels)
print(x_trains)
with open("label.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_trains, np.array(y_labels))
recognizer.save("trainer.yml")
