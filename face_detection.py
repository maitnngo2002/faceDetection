import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

font = cv2.FONT_HERSHEY_SIMPLEX

cascPath = "haarcascade_frontalface_default.xml"
eyePath = "haarcascade_eye.xml"
smilePath = "haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)

# test on a sample image

image = cv2.imread('test.jpg', 0)


plt.figure(figsize=(12, 8))
plt.imshow(image, cmap='gray')
plt.show()

# detect faces

faces = faceCascade.detectMultiScale(
image,
scaleFactor=1.1,
minNeighbors=5,
flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    # draw rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)

plt.figure(figsize=(12, 8))
plt.imshow(image, cmap='gray')
plt.show()


# implement real-time face detection

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # for each face detected, we'll draw a rectangle around the face
    for (x, y, w, h) in faces:
        if w > 250:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            r_gray = gray[y: y + h, x: x + w]
            r_color = frame[y: y + h, x: x + w]
    
    # for each mouth detected, we'll draw a rectangle around it
    smile = smileCascade.detectMultiScale(
        r_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(r_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)
    
    # for each eye detected, draw a rectangle around it

    eyes = eyeCascade.detectMultiScale(r_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(r_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

    cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    video_capture.release()
    cv2.destroyAllWindows()