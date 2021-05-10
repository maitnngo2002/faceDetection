import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
image = cv2.imread('test_image.jpg', 0)

image = np.float32(image) / 255.0

# calculate gradient using Sobel filter

gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

# calculate the direction and magnitude of gradients

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

plt.figure(figsize=(12, 8))
plt.imshow(mag)
plt.show()

# detect face on an image


face_detect = dlib.get_frontal_face_detector()

video_capture = cv2.VideoCapture(0)
flag = 0

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detect(gray, 1)

    for (i, rect) in enumerate(rects):

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()