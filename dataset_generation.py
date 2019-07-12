import numpy as np
import cv2
import os
import argparse

print("""\n\nMajor Project - Face Recognition using Siamese Network (anti-spoofing dataset generation)

by - Abhishek Mann, Kundan Sharma
""")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type = str, required = True, help = "video location")
ap.add_argument("-o", "--output", type = str, required = True, help = "output directory")
ap.add_argument("-s", "--skip", type = int, default = 6, help = "no. of frames to skip")
args = vars(ap.parse_args())

model_config = "deploy.prototxt"
model_l = "res10_300x300_ssd_iter_140000.caffemodel"

nnet = cv2.dnn.readNetFromCaffe(model_config, model_l)

vs = cv2.VideoCapture(args["input"])

read = 0
saved = 0

while True:
    (grabbed_arg, frame) = vs.read()

    if not grabbed_arg:
        break

    read += 1

    if read % args["skip"] != 0:
        continue

    (fr_height, fr_width) = frame.shape[:2]
    inp_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    nnet.setInput(inp_blob)
    net_output = nnet.forward()

    if len(net_output) > 0:
        iter = np.argmax(net_output[0,0,:,2])
        conf_val = net_output[0,0,iter,2]

        if conf_val > 0.5:
            bbox = net_output[0,0,iter,3:7] * np.array([fr_width, fr_height, fr_width, fr_height])
            (leftX, topY, rightX, bottomY) = bbox.astype("int")
            detected_face = frame[topY:bottomY, leftX:rightX]

            fr = os.path.sep.join([args["output"], "{}.png".format(saved)])
            cv2.imwrite(fr, detected_face)
            saved += 1

            print("||Standby|| saved {} to storage".format(fr))

vs.release()
cv2.destroyAllWindows()
