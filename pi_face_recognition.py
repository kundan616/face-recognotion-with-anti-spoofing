import imutils
import time
import os
import cv2
import pickle
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import face_recognition

print("""\n\nMajor Project - Face Recognition using Siamese Network (Face Recogniton + anti-spoofing)

by - Abhishek Mann, Kundan Sharma
""")

print("||Standby|| loading cascade + face embeddings...")
face_embeddings = pickle.loads(open("face_embeddings.pickle","rb").read())
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("||Standby|| loading anti-spoofing network...")
model = load_model("anti_spoof_model.model")
label_encoder = pickle.loads(open("label_encoded.pickle","rb").read())

print("||Standby|| Starting video from camera: ")
vs = VideoStream(0).start()
time.sleep(3.0)

fps = FPS().start()

while True:
    frame = vs.read()
    (fr_height, fr_width) = frame[:2]
    frame = imutils.resize(frame, width = 500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rectangles = face_detector.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 6, minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)

    bboxes = [(y, x + w, y + h, x) for (x, y, w, h) in rectangles]

    encodings = face_recognition.face_encodings(rgb, bboxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(face_embeddings["encodings"], encoding, tolerance = 0.5)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = face_embeddings["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(bboxes, names):
        startY = top
        endY = bottom
        startX = left
        endX = right

        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (32, 32))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)


        preds = model.predict(face)[0]
        j = np.argmax(preds)
        label = label_encoder.classes_[j]


        if label == "real":
            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, endY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)

        elif label == "fake":
            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, endY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 2)


    cv2.imshow("face_recognition (major project)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("||Standby|| time elapsed: {:.2f}".format(fps.elapsed()))
print("||Standby|| FPS (average): {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
