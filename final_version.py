import cv2, time
import numpy as np
from djitellopy import Tello

drone = Tello()
drone.connect()
drone.streamon()
drone.takeoff()

def center_and_distance_calculations(largest_face, frame):
    face_center = None
    frame_center = None

    f_x, f_y,f_w, f_h = largest_face
    face_center = (f_x + f_w // 2, f_y + f_h // 2)
    area = f_h * f_w

            # Get the center coordinates of the frame
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    dis_x = face_center[0] - frame_center[0]
    dis_y = face_center[1] - frame_center[1]

    return f_x, f_y,f_w, f_h, dis_x, dis_y, area


def drone_calculation(distance_x, distance_y, area):
    move_threshold = 20
    fbmin, fbmax = 60000, 70000
    fb, up = 0, 0
    propotional_gain = 0.1

    yaw_speed = distance_x * propotional_gain
    yaw_speed = int(np.clip(yaw_speed, -100, 100))

    if abs(distance_y) > move_threshold:
        if distance_y < 0:
            up = 20
        else:
            up = -20

    if area > fbmin and area < fbmax:
        fb = 0
    elif area > fbmax:
        fb = -20
    elif area < fbmin and area != 0:
        fb = 20

    drone.send_rc_control(0, fb, up, yaw_speed)



def face_tracking():

    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:

        frame = drone.get_frame_read().frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # If a face is detected, track it
        if len(faces) > 0:
            # Get the largest face (assuming it's the closest)
            largest_face = max(faces, key=lambda f: f[2] * f[3])

            # Get the center coordinates of the fac
            f_x, f_y,f_w, f_h, dis_x, dis_y, area = center_and_distance_calculations(largest_face, frame)
            # Draw a rectangle around the face
            cv2.rectangle(
                frame,
                (f_x, f_y),
                (f_x + f_w, f_y + f_h),
                (0, 255, 0),
                2,
            )

            drone_calculation(dis_x, dis_y, area)

        cv2.imshow("Drone Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    drone.streamoff()
    drone.land()
    cv2.destroyAllWindows()

face_tracking()
