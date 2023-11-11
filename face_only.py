import cv2
import time
import numpy as np

fbRange = [6200,6800]
pid = [0.4,0.4,0]
pError = 0
w,h = 360,240
def find_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imggrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imggrey, 1.2, 8)

    myFacelListC = []
    myFaceListArea = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+ w, y+ h), (0,255,0), 2)
        cx = x + w// 2
        cy = y + h// 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0,255,0))
        myFacelListC.append([cx,cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) !=0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFacelListC[i], myFaceListArea[i]]
    else:
         return img, [[0,0],0]

def trackFace(info, w, pid, pError):
    area = info[1]
    x,y = info[0]
    error = x - w//2
    speed = pid[0]*error + pid[1]* (error-pError)
    speed = int(np.clip(speed,-100, 100))
    fb = 0

    if area > fbRange[0] and area <fbRange[1]:
        fb=0
    if area> fbRange[1]:
         fb=-20
    elif area < fbRange[0] and area != 0:
         fb= 20
    
    if x == 0:
         speed = 0
         error = 0
    #me.send_rc_control(0,fb,0,speed)
    print(speed, fb)
    return error


cap = cv2.VideoCapture(0)
while True:
    _,img = cap.read()
    img = cv2.resize(img, (w,h))
    img, info = find_face(img)
    pError = trackFace(info, w, pid, pError)
    cv2.imshow("face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()