import cv2
import os
from datetime import datetime
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

#make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
face_name = input('\n enter user name end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

  
# Parent Directory path
BASE_DIR=os.path.dirname(os.path.abspath('_file_'))
parent_dir = os.path.join(BASE_DIR, "dataset")

  

# Path
path = os.path.join(parent_dir, face_name)
  
if not os.path.isdir(path):
    os.mkdir(path)

# Initialize individual sampling face count
count = 0

#start detect your face and take 50 pictures
while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)    
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(path + '/' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 50: # Take 50 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


