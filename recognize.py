import cv2
import numpy as np
import os
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
facecascade=cv2.CascadeClassifier("D:\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
id=0
names=['Unknown','Rebel']
cam=cv2.VideoCapture(0)
cam.set(3,680)
cam.set(4,480)
minw=0.1*cam.get(3)
minh=0.1*cam.get(3)
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(  gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minw), int(minh)))
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
        if(confidence<100):
            id=names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
    