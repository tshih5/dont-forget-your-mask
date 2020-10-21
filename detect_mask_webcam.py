from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import time
import cv2
import os

def detect_and_predict_mask(frame, face_net, mask_net):
    # grab from dimensions of frame and create blob from it
    (h, w) = frame.shape[:2]
    #use standard ImageNet means
    means = (103.939, 116.779, 123.68)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), means)

    #pass blob through neural network to obtain face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    #initialize arrays
    faces = []
    locs = []
    preds = []
    subjects = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # get face probability
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_0, y_0, x_1, y_1) = box.astype("int")
            #ensure the bounding box does not exceed (w - 2,h - 2) otherwise it will cause a crash
            #(0, 0) represents the top left of the frame
            x_0 = min(max(x_0, 0), w - 2)
            y_0 = min(max(y_0, 0), h - 2)
            x_1 = min(x_1, w - 1)
            y_1 = min(y_1, h - 1)

            #crop frame to detection area and preprocess
            face = process_frame_to_face(frame, (x_0, x_1, y_0, y_1))

            # append face and location data to array
            faces.append(face)
            locs.append((x_0, y_0, x_1, y_1))
            subjects.append(i + 1)

    # predict if at least 1 face detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return (locs, preds, subjects)

def process_frame_to_face(frame, coord):
    face = frame[coord[2]:coord[3], coord[0]:coord[1]]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    return face

if __name__ == '__main__':
    #enable gpu
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)) #device_count = {'GPU': 1}
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    #parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fmodel", type=str, default="face_detector",
                    help="path to detector model directory")
    ap.add_argument("-m", "--mmodel", type=str, default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    
    # load opencv face detector model from disk
    print("[INFO] loading face detector model...")
    
    prototxt_path = os.path.sep.join([args["fmodel"], "deploy.prototxt"])
    weights_path = os.path.sep.join([args["fmodel"], "res10_300x300_ssd_iter_140000.caffemodel"])
    #read model and initialize to variable
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    
    #load the face mask detector .model file from disk
    print("[INFO] loading face mask detector model...")
    mask_net = load_model(args["mmodel"])
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(2.0)

    # loop over the frames from the video stream
    loop_it = True
    while loop_it:
        # grab frame from video
        ret, frame = vs.read()
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (location, predictions, subjects) = detect_and_predict_mask(frame, face_net, mask_net)
        #loop over predictions and corresponding locations
        for (box, prediction, subject) in zip(location, predictions, subjects):
            # get box coordinates
            (x_0, y_0, x_1, y_1) = box
            #get prediction percentages
            (mask, without_mask) = prediction
            #if mask probability is greater than withoutmask probability
            if mask > without_mask:
                label = "Mask"
                #BGR format
                color = (0, 255, 0)
            else:
                label = "No Mask"
                #BGR format
                color = (0, 0, 255)

            #format label string to be put on top of bounding box
            label = "{0}: {1:.2f}%".format(label, max(mask, without_mask) * 100)
            obj_name = "Subject #{0}".format(subject)
            #display the label and bounding box rectangle on the output
            cv2.putText(frame, label, (x_0, y_0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, obj_name, (x_0, y_1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (255, 0, 0), 2)
        
        #display output frame
        cv2.imshow("Frame", frame)
        #wait for q press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[LOG] closing video stream...")
            loop_it = not(loop_it)

    #perform cleanup
    print("[LOG] Performing Cleanup...")
    vs.release()
    cv2.destroyAllWindows()
    print("[LOG] Done.")
