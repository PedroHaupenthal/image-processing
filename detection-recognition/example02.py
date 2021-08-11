import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/detection-recognition/base_files/eu.jpg" -O "eu.jpg"
!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/detection-recognition/base_files/cascades/haarcascade_eye.xml" -O "haarcascade_eye.xml"

img1 = cv.imread("eu.jpg")
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")

res = eye_cascade.detectMultiScale(img2, 
                                   scaleFactor = 1.25, 
                                   minNeighbors = 5)

if res is not None:
  for (x, y, larg, alt) in res:
    img1 = cv.rectangle(img1, 
                        (x, y),
                        (x + larg, y + alt), 
                        (0, 200, 0), 15)

plt.figure(figsize=(15,15))
plt.imshow(img1), plt.axis("off")
plt.show()