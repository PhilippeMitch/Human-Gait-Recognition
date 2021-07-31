# Importing all necessary libraries 
import os 
import cv2
import argparse
import imutils
   
#https://www.geeksforgeeks.org/pedestrian-detection-using-opencv-python/
# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--video", type=str, required=True,
# 	help="path to our input video")
# args = vars(ap.parse_args())
  
# Read the video from specified path
# cam = cv2.VideoCapture(args["video"]) 
cam = cv2.VideoCapture("video/Test/DjakiteFabien.mp4") 

  
try:
    # creating a folder named data 
    if not os.path.exists('test_data'): 
        os.makedirs('test_data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
  
while(True): 
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images
        name = 'test_data/frame' + str(currentframe)
        print ('Creating...' + name)

        frame = imutils.resize(frame,
                       width=min(400, frame.shape[1]))
   
        # Detecting all the regions in the 
        # Image that has a pedestrians inside it
        (regions, _) = hog.detectMultiScale(frame, 
                                            winStride=(4, 4),
                                            padding=(3, 3),
                                            scale=1.2)
  
        # writing the extracted images 
        # Drawing the regions in the Image
        idx = 0
        for (x, y, w, h) in regions:
            # cv2.rectangle(image, (x, y), 
            #               (x + w, y + h), 
            #               (0, 0, 255), 2)
            if w>60 and h>60:
                idx+=1
                new_img=frame[y:y+h,x:x+w]
                #cropping images
                cv2.imwrite(name + '.png', new_img)

        
        # cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()