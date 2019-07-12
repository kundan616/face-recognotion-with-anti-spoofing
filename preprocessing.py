import imutils
from imutils import paths
import face_recognition
import pickle
import os
import cv2

print("""\n\nMajor Project - Face Recognition using Siamese Network (embedding generation)

by - Abhishek Mann, Kundan Sharma
""")

print("\n\n||Standby|| parsing faces directory")

image_paths = list(paths.list_images("faces_dataset"))

known_encodings = []
known_names = []

for (iter, image_path) in enumerate(image_paths):
    name = image_path.split(os.path.sep)[-2]
    print("||Standby|| processing image of " + str(name) + " {}--{}".format(iter+1, len(image_paths)))

    img = imutils.resize(cv2.imread(image_path), width = 600)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = face_recognition.face_locations(rgb, model = "cnn")

    generated_encodings = face_recognition.face_encodings(rgb, bboxes)

    for encoding in generated_encodings:
        print("\n\n\n")
        print(encoding)
        known_names.append(name)
        known_encodings.append(encoding)


print("\n\n||Standby|| serializing facial embeddings...")
encodings_data = {"encodings": known_encodings, "names": known_names}
f = open("face_embeddings.pickle", "wb")
f.write(pickle.dumps(encodings_data))
f.close()
