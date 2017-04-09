__author__ = 'ankush'

import cv2

videoCapture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

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
    face = faceCascade.detectMultiScale(
        clahe_image,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if len(face) == 1:
        faceSlice = crop_face(clahe_image,face)
        cv2.imshow("Face",faceSlice)
    else:
        print("no/multi face detected, passing frame")

    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('webcam',frame)

def crop_face(gray,face):
    for (x,y,w,h) in face:
        faceSlice = gray[y:y+h,x:x+w]
        cropped_face = cv2.resize(faceSlice,(150,150))
    return cropped_face



while True:
    detect_face()