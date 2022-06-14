'''---------------------------------------------------
   Program  : HandWrittenDigitRecognition.py; 
   Version  : 1.0;
   Date     : April 12, 2022
   Modified by : Nancy Saxena 
   Ref : https://github.com/hualili/opencv/tree/master/deep-learning-2022s
-------------------------------------------------------''' 
######Import necessary Library
from tensorflow import keras 
from tensorflow.keras.models import load_model
import sys
import cv2
import numpy as np
import time 
import os

###Define prediction for image
def display_pred(contours,Mnist_model,G_Image,img):    
    for i in range(len(contours)):
        [x, y, weight, height] = cv2.boundingRect(contours[i])
        if(height>10):
            padding =15
            Cropped_img = G_Image[y-padding:y+height+padding, x-padding:x+weight+padding]
            #######Preprocessing  the cropped image     
            Cropped_img =Cropped_img/255.0
            #######Code to aspect ratio
            
            if (height>weight):
                padding_h = (height-weight)//2
                Cropped_img1 = cv2.copyMakeBorder(Cropped_img, 0, 0,padding_h, padding_h, cv2.BORDER_CONSTANT, None, 0)
            else:
                padding_h = (weight-height)//2
                Cropped_img1 = cv2.copyMakeBorder(Cropped_img, padding_h, padding_h,0,0, cv2.BORDER_CONSTANT, None, 0)
                
            #####Resize the cropped image from 28,28
            Cropped_img = cv2.resize(Cropped_img, (28,28))
            
            #####Model prediction 
            digit_preds = Mnist_model.predict(Cropped_img.reshape(1,28, 28, 1)).argmax()
            
            #######Defining ROI
            cv2.rectangle(img, (x, y), (x + weight, y + height), (0, 0, 0), 2)
            print(digit_preds)
            prediction_name= str(digit_preds)
            cv2.imshow(prediction_name,Cropped_img1)
            digit_preds = str(digit_preds) 
            cv2.putText(img, digit_preds, (x, y-30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 0),1)            
    return img

def video_feed_live():                                                                                   
    print("Capturing video from live webcam feed")
    
    #####Capture the webcam 
    cap = cv2.VideoCapture(0)
    
    #####Check if the video feed open
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
    
    #####For the sake of restricting the time for the video capture
    start_time = time.time()
    
    #####Load the model
    Mnist_model = load_model('cnn_model.h5')
    out = cv2.VideoWriter('out_live.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (256,256))

    #####define countours
    while(int(time.time()-start_time) < 200):
        ret, frame = cap.read()        
        if ret == True:
            img = cv2.resize(frame, (256,256))
            G_Image = gray_coversion(img)
            ####Defining edges
            edges = cv2.Canny(G_Image, 100, 200)
            ####Defining contours
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ####Display prediction
            display_pred(contours,Mnist_model,G_Image,img)
            cv2.imshow('frame',img)
            out.write(img)
            if cv2.waitKey(1) == 27:
                break
    #####Break the loop
        else:
            break                                                                              


        
   
def stored_video_feed():
    # Check if camera opened successfully
    print("Enter the full path to the file")
    filepath = input()
    cap = cv2.VideoCapture(filepath)
    assert os.path.exists(filepath), "File not found at, "+str(filepath)
    
    ####Check if file exists:    
    
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer. 
    target_size =(256,256)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('out_recorded.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5 , target_size)
    ####Recording time to read the file 
    start_time = time.time()
    ####Load the file 
    Mnist_model = load_model('cnn_model.h5')
    while(cap.isOpened() and time.time()-start_time < 20):
        ret, frame = cap.read()
        if ret == True: 
            img = cv2.resize(frame, (256,256))
            G_Image = gray_coversion(img)
            ####Define edges 
            edges = cv2.Canny(G_Image, 100, 200)
            ####Define contours 
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display_pred(contours,Mnist_model,G_Image,img)
            out.write(img)
            cv2.imshow('frame',img)
    ####Press Q on keyboard to stop recording
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
    ####Break the loop
        else:
            break  
            
def gray_coversion(img):
    #### Frame coversion to gray image
    G_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    G_Image = cv2.GaussianBlur(G_Image, (5, 5), 0)
    G_Image = cv2.adaptiveThreshold(G_Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    return G_Image
   
def main():
    import warnings
    warnings.filterwarnings("ignore", message="Could not load dynamic library")
    print("Handwritten digit recognition:Version-1.0")
    print("Type 1: For live video feed, Type 2: For stored video feed")   
    option = input()
    if(option == "1"):
        video_feed_live();
    elif(option == "2"):
        stored_video_feed();
    else:
        print("Please enter  1: For live video feed, Type 2: For stored video feed, Rerun the script")       
   
        
if __name__ == "__main__":
    main()

