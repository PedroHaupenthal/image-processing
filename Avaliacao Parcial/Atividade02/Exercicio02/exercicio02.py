# Exercicio 02
# Pedro Afonso Ferreira Haupenthal 823974

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/Atividade02/Exercicio02/laranjas.jpg" -O "laranjas.jpg"

img1 = cv.imread("laranjas.jpg")
img2 = img1.copy()

img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
img2 = cv.medianBlur(img2, 5)

circles = cv.HoughCircles(img2, 
                           cv.HOUGH_GRADIENT, 
                           1,
                           120, 
                           param1 = 150,
                           param2 = 30,
                           minRadius = 60,
                           maxRadius = 0)

circles = np.uint16(np.around(circles))

if circles is not None:
  print("Qtde de circulos: %i" %(len(circles[0,:])))

  for circle in circles[0,:]:
    cv.circle(img1, (circle[0], circle[1]), circle[2], (255,255,0), 5)

plt.figure(figsize=(25,25))
plt.subplot(121), plt.imshow(img1)
plt.subplot(122), plt.imshow(img2, cmap='gray')