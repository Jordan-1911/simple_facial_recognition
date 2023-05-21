import cv2
import pathlib

import cv2
import os

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
# print(cascade_path)  # sanity check to check file path

def detect_from_video(cascade_path):

    clf = cv2.CascadeClassifier(str(cascade_path))

    # camera = cv2.VideoCapture(0)  # number may be different depending on how many cameras your machine has. zero for one camera
    camera = cv2.VideoCapture("elon_musk.mp4")

    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = clf.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # plot rectangle around face
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)  # in BGR
            
        
        # show image
        cv2.imshow("Faces", frame)
        if cv2.waitKey(1) == ord("q"):  # if Q is pressed, break
            break

    camera.release()
    cv2.destroyAllWindows()


# detect_from_video(cascade_path)  # call video facial recognition function

# we can also use cv2 for images as seen below
def detect_from_image(cascade_path):
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    image = cv2.imread('./images/musk.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # in BGR
        
    cv2.imshow('Face', image)
    cv2.waitKey()
    
# detect_from_video(cascade_path)
detect_from_image(cascade_path)

