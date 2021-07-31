In human identification, the process of identifying a person by gait is an emerging research trend in the field of visual surveillance and the only way to identify a person from a distance. Identifying a person by gait is a type of biometric technology, it has recently been used to recognize a person by the style of their walk.

This work first reviews the different techniques developed over the past decades to better understand the subject, then presents our solution approach based on the use of images to represent walking, such as the Gait Energy Image (GEI ) to create the model.

We start with the conventional approach to gait recognition which includes feature extraction and classification using SVM. Feature vectors for classification were constructed using dimension reduction on cumulants calculated by principal component analysis (PCA). Then a second Deep Learning model where the representations of walking are used as input into a convolutional neural network, which is used to perform classification or to extract a feature vector which is then classified using methods of machine learning to create our second model.

The models are trained and tested with a dataset that we have built, containing seven (7) people in a 180 degree angle.
After implementation and experimentation, the results are interesting, on the other hand open to the necessary improvements. 

#Extraire des images d'une vidéo

```python
# Importing all necessary libraries 
import os 
import cv2
import argparse
import imutils

# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", type=str, required=True,
	help="path to our input video")
args = vars(ap.parse_args())

# Read the video from specified path
cam = cv2.VideoCapture(args["video"]) 

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
```
