import cv2
import numpy as np
import time
from djitellopy import Tello


def face_tracking():
    """
    Function to make DJI Tello drone follow a user's face using computer vision.

    This function uses OpenCV to detect and track a user's face in real-time video feed from the drone's camera.
    The drone adjusts its position to keep the user's face in the center of the frame.

    Note:
    - This function assumes that the DJI Tello drone is connected and ready to fly.
    - Make sure to install the required dependencies: cv2, numpy, and djitellopy.

    Returns:
    - None
    """

    # Initialize the Tello drone
    drone = Tello()

    # Connect to the drone
    drone.connect()

    # Start the video stream
    drone.streamon()
    drone.takeoff()

    # Create a window to display the video feed

    # Initialize the face detection cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Variables for face tracking
    face_center = None
    frame_center = None
    move_threshold = 20
    fbmin = 60000
    fbmax = 70000
    fb = 0
    up = 0

    # Main loop for face tracking
    while True:
        # Get the current frame from the video stream
        frame = drone.get_frame_read().frame

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # If a face is detected, track it
        if len(faces) > 0:
            # Get the largest face (assuming it's the closest)
            largest_face = max(faces, key=lambda f: f[2] * f[3])

            # Get the center coordinates of the face
            face_x, face_y, face_w, face_h = largest_face
            face_center = (face_x + face_w // 2, face_y + face_h // 2)
            area = face_h * face_w

            # Get the center coordinates of the frame
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

            # Draw a rectangle around the face
            cv2.rectangle(
                frame,
                (face_x, face_y),
                (face_x + face_w, face_y + face_h),
                (0, 255, 0),
                2,
            )

            # Calculate the distance between the face center and frame center
            distance_x = face_center[0] - frame_center[0]
            distance_y = face_center[1] - frame_center[1]

            # Move the drone based on the distance from the face center to the frame center
            # if abs(distance_x) > move_threshold:
            # if distance_x < 0:
            # drone.send_rc_control(-20, 0, 0, 0)
            # else:
            # drone.send_rc_control(20, 0, 0, 0)
            yaw_speed = distance_x * 0.1

            print(f"face_x {face_x} frame_center_x {frame_center[0]}")
            print(distance_x)
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

        # Display the frame with face tracking
        cv2.imshow("Drone Face Tracking", frame)

        # Check for key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Stop the video stream
    drone.streamoff()

    # Land the drone
    drone.land()

    # Close the window
    cv2.destroyAllWindows()


# Call the face_tracking function to start the drone face tracking
face_tracking()
