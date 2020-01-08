import numpy as np
import cv2
face_id=input("Enter name")
face_cascade=cv2.CascadeClassifier("D:\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
count=0
while(True):
    ret,img=cap.read()
   # frame=cv2.flip(frame,-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
    for(x,y,w,h) in faces:
        count+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color=img[y:y+h,x:x+w]
        roi_gray=gray[y:y+h,x:x+w]
        gray=cv2.resize()
        cv2.imwrite("dataset/user."+str(face_id)+"."+str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)
    #cv2.imshow('gray',gray)
   #1 print(roi_color)
    k=cv2.waitKey(30)&0xff
    if(k==27):
        break
cap.release()
cv2.destroyAllWindows()    

