# Handwritten-Digit-Recognition-Deep-Learning

## System requirements that are satisfied:
a.The input video is hand written by self
b . The captured video is in color
c. The bounding boxes are being displayed individually over each digit
d. Preprocessed image cropped digits are displayed before feeding it to CNN
e. Live video feeds are displaying the bounding boxes and predictions in real time.

## How to Run the code:
2.1 Package/Library requirements:
A. Opencv version: 4.5.5
B.Keras version: 2.8.0
C.Tensorflow version : 2.8.0
D.Python version : 3.8.8
2.2 Prediction model:
CNN: Convolutional neural network with :
C1 C2 M1 C2 C3 M2 .
The model accuracy is approx 96%
The cnn.h5 file is present in the Github 

## main.py code:
Live video:
Command : python main.py
Choose option 1
Press enter
Sample output: out_live.avi

Stored Video:
Command:python main.py
Choose option 2
Put the full file path
Press enter.

References:  https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python

             https://github.com/hualili/opencv/tree/master/deep-learning-2022s

             https://stackoverflow.com/questions/60869306/how-to-simple-crop-the-bounding-box-in-python-opencv#:~:text=I%20figured%20it%20out%20the%20formula%20for%20cropping,print%20%28%5BX%2CY%2CW%2CH%5D%29%20plt.imshow%20%28cropped_image%29%20cv2.imwrite%20%28%27contour1.png%27%2C%20cropped_image%29%20Share
            
         
            
