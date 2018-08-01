# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 23:52:35 2018

@author: pardeshibabu
"""

import numpy as np
#import matplotlib.pyplot as plt
import cv2
import sqlite3

faceDetect = cv2.CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)


def insert(Id,Name):
    conn = sqlite3.connect("faceBace.db")
    cmd = "SELECT * FROM people WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        if(isRecordExist==1):
            cmd="UPDATE people SET Name="+str(Name)+"WHERE ID="+str(Id)
        else:
            cmd= "INSERT INTO people(ID,Name) Values("+str(Id)+","+str(Name)+")"
        conn.execute(cmd)
        conn.commit()
        conn.close()




id = input('enter user id: ')
name = input('enter name: ')
insert(id,name)
sampleNum=0;
             
while(True):
    ret,img=cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    #faces = face_cascade.detectMultiScale(gray,1.3,5)
    #print(faces)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;    
        cv2.imwrite("F:/DataSet/user."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(255,0,0),2)
        cv2.waitKey(100);
        
    
    cv2.imshow("face",img)
    cv2.waitKey(100)
    if(sampleNum>20):
        break

cam.release()
cv2.destroyAllWindows()    
