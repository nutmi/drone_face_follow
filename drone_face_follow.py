import cv2
import time
import numpy as np

fbRange = [60000,70000]

def find_face(img):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imggrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imggrey, 1.2, 8)
    area = 0
    cx = 0

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+ w, y+ h), (0,255,0), 2)
        cx = x + w// 2
        cy = y + h// 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0,255,0))
    
    return img, area, cx



def track_face(center_face_x, center_frame_x):
     error_x = center_face_x - center_frame_x
     return error_x

cap = cv2.VideoCapture(0)
while True:
    _,img = cap.read()
    h,w,_ = img.shape
    center_x = w//2
    img, area, cx= find_face(img)
    error_x = track_face(cx, center_x)
    print(f"area: {area}, center_x: {center_x}, cx: {cx}, error_x: {error_x}")
    print(np.clip(error_x, -100, 100))
    cv2.imshow("face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()