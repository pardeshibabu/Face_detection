import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3

rec=cv2.createLBPHFaceRecognizer();
rec.load("F:\\recognizer\\trainningData.yml")
faceDetect=cv2.CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
path= 'F:\\DataSet'



def getinfo(id):
        conn=sqlite3.connect("F:\\FaceBase.db")
        cmd= "SELECT * FROM People WHERE ID="+str(id)
        cursor=conn.execute(cmd)
        info=None
        for row in cursor:
                #print(row)
                info=row
        conn.close()
        return info


cam = cv2.VideoCapture(0);
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):
        ret,img = cam.read();
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,1.3,5);
        for (x,y,w,h) in faces:
                id,conf = rec.predict(gray[y:y+h,x:x+w])
                #print(conf)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                info = getinfo(id)
                #print(info)
                if(info!= None):
                        cv2.cv.PutText(cv2.cv.fromarray(img),str(info[1]),(x,y+h),font,255);               
                        
        cv2.imshow("face",img);
        if(cv2.waitKey(10)==ord('q')):
                break;

cam.release()
cv2.destroyAllWindows()






















   

		

 	

