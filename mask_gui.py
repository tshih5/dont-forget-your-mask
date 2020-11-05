import tkinter as tk
from PIL import Image, ImageTk
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client
import tensorflow as tf
from datetime import datetime, date
import numpy as np
import cv2
import argparse
import time
import os
import re

class TwilioClient():
    def __init__(self, account_sid, auth_token):
        self.client = Client(account_sid, auth_token)
        #initalize default numbers
        self.twilio_number=""
        self.personal_number=""
    
    def send_sms_message(self, message_text):
        message = self.client.messages.create(body = message_text, from_=self.twilio_number, to=self.personal_number)

    def set_twilio_number(self, number):
        
        self.twilio_number = number
    
    def get_twilio_number(self):
        return self.twilio_number

    def set_personal_number(self, number):

        self.personal_number = number
    
    def get_personal_number(self):
        return self.personal_number

    


class Application(tk.Frame):
    def __init__(self, window, window_title, video_source=0):
        super().__init__(window)
        self.window = window
        self.window.title(window_title)
        
        self.start_time = datetime.now()
        self.curr_time = datetime.now()
        self.sms_print_times = 0
        # initialize video source
        self.interval = 20
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)

        # bool flag for video processing start/stop
        self.process_video = False

        # create canvas
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, columnspan=2)

        # add phone number entry
        tk.Label(window, text="Phone Numbers (include country code)").grid(row=1)
        
        # Add default personal phone number 
        tk.Label(window, text="Personal Number").grid(row=2)
        self.entry_personal_number = tk.Entry(window)
        self.entry_personal_number.insert(0, os.getenv("PERSONAL_NUMBER"))
        self.entry_personal_number.grid(row=2, column=1)
        
        # Add default Twilio Number 
        tk.Label(window, text="Twilio Number").grid(row=3)
        self.entry_twilio_number = tk.Entry(window)
        self.entry_twilio_number.insert(0, os.getenv("TWILIO_NUMBER"))
        self.entry_twilio_number.grid(row=3, column=1)

        #show warning to user
        self.label_valid_warn = tk.Label(window, text="if numbers are blank, they are invalid and must be re-entered",font=("helvetica", 10))
        self.label_valid_warn.grid(row=5, columnspan=2)

        # add blank label for number validation
        self.label_personal_number = tk.Label(window, text= "Phone number is: ",font=("helvetica", 10))
        self.label_personal_number.grid(row=6, column=1)
        self.label_twilio_number = tk.Label(window, text= "Twilio number is: ",font=("helvetica", 10))
        self.label_twilio_number.grid(row=6, column=0)

        # create Twilio Client
        self.tw_client = TwilioClient(os.getenv("ACCOUNT_SID"), os.getenv("ACCOUNT_AUTH"))

        # button to set phone numbers
        self.btn_set_vars = tk.Button(window, text="Get numbers", width=30, command=self.set_vars)
        self.btn_set_vars.grid(row=4, column=1)

        #button to toggle stream processing
        self.btn_proccess_stream = tk.Button(window, text= "Start Net", width=30, command=self.toggle_process_stream)
        self.btn_proccess_stream.grid(row=4, column=0)
        self.update_image()

    def set_vars(self):
        pn = self.entry_personal_number.get()
        tn = self.entry_twilio_number.get()

        #check validity of both numbers
        pn_found, pn_result = self.is_valid_number(pn)
        tn_found, tn_result = self.is_valid_number(tn)
        # set numbers in twilio client
        self.tw_client.set_personal_number(pn_result)
        self.tw_client.set_twilio_number(tn_result)
        # update label
        self.label_personal_number.config(text= "Phone number is: " + self.tw_client.get_personal_number())
        self.label_twilio_number.config(text= "Twilio number is: " + self.tw_client.get_twilio_number())
    
    # determines whether the input phone number is valid or not.
    def is_valid_number(self, number):
        re_string = "\+?[.\s()\-]{0,3}\d{1,2}[.\s()\-]{0,3}\d{3}[.\s()\-]{0,3}\d{3}[.\s()\-]{0,3}\d{4}"
        phone_regex = re.compile(re_string)
        found = phone_regex.fullmatch(number)
        result = ""

        if found:
            print(number, "looks like a valid number!\n")
            result = "+" + "".join(filter(lambda i: i.isdigit(), number))
        else:
            print(number, "is not a valid number. Please try again\n")
        
        return found, result

    # toggle stream processing on and off
    def toggle_process_stream(self):
        self.process_video = not self.process_video
        if self.process_video:
            print("Stream start! make sure your contact numbers are saved!")
            self.btn_proccess_stream.config(text="Stop Net")
            # when stream processing starts initialize start time
            self.start_time = datetime.now()
        else:
            print("Stream Stopped.")
            self.btn_proccess_stream.config(text="Start Net")

    # update window with frame every interval ms
    def update_image(self):
        ret, self.image = self.vid.get_frame(self.process_video)
        self.curr_time = datetime.now()
        if ret:
            self.image = Image.fromarray(self.image) # to PIL
            self.image = ImageTk.PhotoImage(self.image) # to ImageTk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # determine wheter to print sms status
        if (self.curr_time - self.start_time).total_seconds() > 20 and self.sms_print_times == 0:
            print("ready to send sms")
            self.sms_print_times += 1

        if max(self.vid.frame_counts) > 55 and (self.curr_time - self.start_time).total_seconds() > 20:
            # reset frame count
            self.vid.frame_counts = [0] * 5
            self.sms_print_times = 0
            print("sending angry text...")
            self.start_time = datetime.now()
        self.window.after(self.interval, self.update_image)


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # get command line arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--fmodel", type=str, default="face_detector",
                    help="path to detector model directory")
        ap.add_argument("-m", "--mmodel", type=str, default="mask_detector.model",
                    help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
        self.args = vars(ap.parse_args())

        # load pre trained neural nets
        print("[LOG] loading face detector...")
        self.face_net = self.load_face_net(self.args["fmodel"])
        print("[LOG] loading mask detector...")
        self.mask_net = load_model(self.args["mmodel"])

        # initialize frame counts
        self.frame_counts = [0] * 5
    
    # cleanup
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
    
    def detect_faces(self, frame):
        # grab from dimensions of frame and create blob from it
        (h, w) = frame.shape[:2]
        # use standard ImageNet means for color channels
        means = (103.939, 116.779, 123.68)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), means)

        # pass blob through neural network to obtain face detections
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        return (h, w, detections)

    def process_frame_to_face(self, frame, coord):
        face = frame[coord[2]:coord[3], coord[0]:coord[1]]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        return face

    def detect_and_predict_mask(self, frame):
        # initialize lists
        faces = []
        locs = []
        preds = []
        subjects = []

        # obtain face detections
        h, w, detections = self.detect_faces(frame)

        # loop over the face detections and append face data to lists
        for i in range(0, detections.shape[2]):
            # get face probability
            confidence = detections[0, 0, i, 2]
            # filter out weak face detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x_0, y_0, x_1, y_1) = box.astype("int")
                # ensure the bounding box does not exceed (w - 2,h - 2) otherwise it will cause a crash
                # (0, 0) represents the top left of the frame
                x_0 = min(max(x_0, 0), w - 2)
                y_0 = min(max(y_0, 0), h - 2)
                x_1 = min(x_1, w - 1)
                y_1 = min(y_1, h - 1)

                # crop frame to detection area and preprocess
                face = self.process_frame_to_face(frame, (x_0, x_1, y_0, y_1))

                # append face and location data to array
                faces.append(face)
                locs.append((x_0, y_0, x_1, y_1))
                subjects.append(i + 1)

        # find prediction for each face detected
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.mask_net.predict(faces, batch_size=32)

        return (locs, preds, subjects)

    #return source frame
    def get_frame(self, process=False):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                if not process:
                    self.frame_counts = [0] * 5
                    # return raw frame converted to rgb
                    return(ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    #Process video frame and return altered image
                    location, predictions, subjects = self.detect_and_predict_mask(frame)
                    for index, (box, prediction, subject) in enumerate(zip(location, predictions, subjects)):
                        # get box coordinates
                        (x_0, y_0, x_1, y_1) = box
                        # get prediction percentages
                        (mask, without_mask) = prediction
                        # if mask probability is greater than withoutmask probability
                        if mask > without_mask:
                            label = "Mask"
                            # BGR format
                            color = (0, 255, 0)
                            self.frame_counts[index] = 0
                        else:
                            label = "No Mask"
                            # BGR format
                            color = (0, 0, 255)
                            self.frame_counts[index] += 1

                        # format label string to be put on top of bounding box
                        label = "{0}: {1:.2f}%".format(label, max(mask, without_mask) * 100)
                        obj_name = "Subject #{0}".format(subject)
                        # display the label and bounding box rectangle on the output
                        cv2.putText(frame, label, (x_0, y_0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        cv2.putText(frame, obj_name, (x_0, y_1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (255, 0, 0), 2)
                    return(ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return(ret, None)
        else:
            return(False, None)

    def load_face_net(self, face_directory):
        prototxt_path = os.path.sep.join([face_directory, "deploy.prototxt"])
        weights_path = os.path.sep.join([face_directory, "res10_300x300_ssd_iter_140000.caffemodel"])
        return cv2.dnn.readNet(prototxt_path, weights_path)

    



if __name__ == "__main__":
    # load environment variables
    load_dotenv()
    root = tk.Tk()
    my_app = Application(root, "Don\'t Forget Your Mask!" )
    my_app.mainloop()