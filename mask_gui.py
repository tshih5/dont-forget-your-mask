import tkinter as tk
from PIL import Image, ImageTk
import cv2
from dotenv import load_dotenv
import os

class Application(tk.Frame):
    def __init__(self, window, window_title, video_source=0):
        super().__init__(window)
        self.window = window
        self.window.title(window_title)
        
        # initialize video source
        self.interval = 20
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)

        # create canvas
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, columnspan=2)

        tk.Label(window, text="Phone Numbers (in the form \'+1-123-456-7890\', No \'-\'s)").grid(row=1)
        tk.Label(window, text="Personal Number").grid(row=2)
        self.personal_number = tk.Entry(window)
        self.personal_number.insert(0, os.getenv('PERSONAL_NUMBER'))
        self.personal_number.grid(row=2, column=1)
        tk.Label(window, text="Twilio Number").grid(row=3)
        self.twilio_number = tk.Entry(window)
        self.twilio_number.insert(0, os.getenv('TWILIO_NUMBER'))
        self.twilio_number.grid(row=3, column=1)
        
        self.btn_sayhi = tk.Button(window, text="Get numbers", width=50, command=self.get_vars)
        self.btn_sayhi.grid(row=4, column=1)
        self.update_image()

    def get_vars(self):
        pn = self.personal_number.get()
        tn = self.twilio_number.get()
        label3 = tk.Label(self.window, text= 'Phone number is: ' + pn,font=('helvetica', 10)).grid(row=5, column=0)
        label4 = tk.Label(self.window, text= 'Twilio number is: ' + tn,font=('helvetica', 10)).grid(row=5, column=1)
        #self.canvas.create_window(400, 400, window=label3)
    
    # update window with frame every interval ms
    def update_image(self):
        ret, self.image = self.vid.get_frame()
        if ret:
            self.image = Image.fromarray(self.image) # to PIL
            self.image = ImageTk.PhotoImage(self.image) # to ImageTk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        self.window.after(self.interval, self.update_image)

    def say_hi(self):
        print("hi there, everyone!")

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # cleanup
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
    
    #return source frame
    def get_frame(self, process=0):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                if process == 0:
                    # return raw frame converted to rgb
                    return(ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    #Process video frame and return altered image
                    print("hi")
            else:
                ret(ret, None)
        else:
            return(False, None)

if __name__ == "__main__":
    load_dotenv()
    root = tk.Tk()
    my_app = Application(root, "Test Window" )
    my_app.mainloop()