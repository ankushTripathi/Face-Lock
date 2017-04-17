
import cv2
import numpy as np
import argparse
import time
import glob
import os
import Update_Model


videoCapture = cv2.VideoCapture(0)



faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye_tree_eyeglasses.xml")

lbphFace = cv2.createLBPHFaceRecognizer()
try:
    lbphFace.load("xml/faces.xml")
except:
    print("no xml found. train first")

parser = argparse.ArgumentParser(description="Options for face recognition")
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true")
args = parser.parse_args()

facedict = {}

people = ["ankush","unknown"]

def grab_webcame_frame():
    ret,frame = videoCapture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit = 2.0,
        tileGridSize = (8,8)
         )
    clahe_image = clahe.apply(gray)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        quit()
    return clahe_image,frame

def detect_face():
    clahe_image,frame = grab_webcame_frame()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(
        clahe_image,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if len(face) == 1:
        faceSlice = crop_face(clahe_image,face)
        cv2.imshow("Face",faceSlice)
    else:
        print("no/multi face detected, passing frame")

#    for (x,y,w,h) in face:
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    return frame

def crop_face(gray,face):
    for (x,y,w,h) in face:
        faceSlice = gray[y:y+h,x:x+w]
        cropped_face = cv2.resize(faceSlice,(150,150))
        facedict["face%s"%(len(facedict)+1)] = cropped_face
    return cropped_face

def check_folders(people):
    for x in people:
        if os.path.exists("dataset\\%s"%x):
            pass
        else:
            os.makedirs("dataset\\%s"%x)

def save_face(person):
    print("\n\n"+ person+" your turn... ")
    for i in range(0,5):
        print(5-i)
        time.sleep(1)

    while len(facedict.keys())<20:
        detect_face()

    for x in facedict.keys():
        cv2.imwrite("dataset\\%s\\%s.jpg"%(person,len(glob.glob("dataset\\%s\\*"%person))),facedict[x])

    facedict.clear()

def update_model(people):
    print("Update Mode active...")
    check_folders(people)
    for i in range(0,len(people)):
        save_face(people[i])
    print("Images saved! Now updating emotion classifier..")
    Update_Model.update(people)
    print("Done!")

def recognize_face(frame):
    predictions = []
    confidence = []
    final_face=[]
    for x in facedict.keys():
        pred,conf = lbphFace.predict(facedict[x])
        predictions.append(pred)
        confidence.append(conf)
        recognized_face = people[max(set(predictions),key=predictions.count)]
        final_face.append(recognized_face)
        level = float("{0:.2f}".format(max(confidence)*0.1))
        print("you're %s"%recognized_face)
    get_face = max(set(final_face),key=final_face.count)
    if  get_face == "ankush":
        face1 = facedict['face2']
        eyes1 = eyeCascade.detectMultiScale(face1,
                                           scaleFactor=1.1,
                                           minNeighbors=20,
                                           minSize=(7,7),
                                           flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        if len(eyes1) > 0:
            face2 = facedict['face5']
            eyes2 = eyeCascade.detectMultiScale(face2,
                                           scaleFactor=1.1,
                                           minNeighbors=20,
                                           minSize=(7,7),
                                           flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            if len(eyes2) == 0:
                print("detected")
                cv2.putText(frame,"welcome mothafucka",(200,300),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    cv2.putText(frame,get_face,(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
 #   cv2.putText(frame,str(level)+"%",(10,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
    cv2.imshow("webcam",frame)
    facedict.clear()

while True:
    frame = detect_face()
    if args.update:
        update_model(people)
        break
    elif len(facedict) == 6:
        recognize_face(frame)
