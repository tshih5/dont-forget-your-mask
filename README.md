# Mask Detector

## Disclaimer
Utilizes some of Adrian Rosebrock's code from [COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/) and adds improvements to familiarize myself with machine learning practices in a fun little project.

## Requirements
* python 3.7
* tensorflow (`pip install tensorflow`)
* NumPy (`pip install numpy`)
* scikit-learn (`pip install scikit-learn`)
* imutils (`pip install imutils`)
* MatplotLib (`pip install matplotlib`)
* openCV (`pip install opencv-contrib-python`)
* Pillow (`pip install Pillow`) (mask_gui.py)
* Twilio (`pip install twilio`)
* A Twilio account
* If on Windows, python-dotenv (`pip install python-dotenv`)
* If using a virtual environment (recommended), `pip install -r requirements.txt`

## Dataset
I used a subset of the [Real World Masked Face (RFMD)](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) data set aggregated with the without_mask portion of Prajna Bhandary's [data set](https://github.com/prajnasb/observations/tree/master/experiements/data) to train my model.

## Files

### `dont_forget_your_mask.py`
Main code. Sends you a text message from your twilio number to your personal number (set in an `.env` file) if it detects you without a mask for long enough

### `mask_gui.py`
Second version of `dont_forget_your_mask.py` that uses tkinter to make a gui for the application. Requires Pillow.

### `train_mask_detector.py`
Trains the mask detector model and outputs a .model file to be used in the main code.

### `train_mask_detector_v2.py`
V2 of `train_mask_detector.py`. Easier on memory as it does not try to load the entire dataset into RAM (Permits user to train on large datasets).
