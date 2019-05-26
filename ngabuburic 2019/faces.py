import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name" : 1}

with open("label.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi_color = frame[x:x+w, y:y+h]
        roi_gray = gray[x:x+w, y:y+h]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            print(id_)

            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2

            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        image_items = 'image.jpg'
        cv2.imwrite(image_items, roi_color)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
