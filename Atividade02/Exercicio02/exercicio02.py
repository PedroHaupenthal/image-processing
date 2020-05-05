# Exercicio 02
# Pedro Afonso Ferreira Haupenthal 823974

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/Atividade02/Exercicio01/pista.jpg" -O "pista.jpg"

img1 = cv.imread("pista.jpg")
img2 = img1.copy()

img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
img2 = cv.Canny(img2, 600, 220)

lines = cv.HoughLinesP(img2, 
                        1, 
                        1*np.pi/90, 
                        50, 
                        minLineLength = 10, 
                        maxLineGap = 100)

if lines is not None:
  print("linhas = %i" %(len(lines)))

  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 5)

plt.figure(figsize=(35,35))
plt.subplot(121), plt.imshow(img1,  cmap='gray')