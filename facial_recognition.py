import cv2
import pathlib

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascades/haarcascade_frontalface_default.xml"
print(cascade_path)  # provided by opencv-python

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)  # number may be different depending on how many cameras your machine has. zero for one camera

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

