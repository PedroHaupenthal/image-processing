# Exercicio 02
# Pedro Afonso Ferreira Haupenthal 823974

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math 

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/Atividade01/Exercicio02/moedas_105.jpg" -O "moedas.jpg"

img = cv.imread("moedas.jpg")
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_binary = img.copy()

color_contour = (255, 0, 0)
thickness_contour = 2

for lin in range(img.shape[0]):
  for col in range(img.shape[1]):
    if(img.item(lin, col) > 50):
      img_binary.itemset((lin, col),0)
    else: 
      img_binary.itemset((lin, col),255)
    
contours, ordem = cv.findContours(img_binary, 
                                  cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)

radius_list = []

for c in contours:
  x, y, w, h = cv.boundingRect(c)
  cv.rectangle(img_binary,
               (x,y),
               (x+w, y+h),
               color_contour,
               thickness_contour)

  area = cv.minAreaRect(c)
  box = cv.boxPoints(area) 
  box = np.int0(box)

  (x, y), radius = cv.minEnclosingCircle(c)
  centro = (int(x),
            int(y))
  radius = int(radius)
  radius_list.append(radius)
  
area_1 = round(math.pi * (radius_list[4] * radius_list[4]), 2)
area_2 = round(math.pi * (radius_list[5] * radius_list[5]), 2)


print(f"Área figura 1 = {area_1}")
print(f"Área figura 2 = {area_2}")
plt.figure(figsize=(15,15))
plt.imshow(img_binary)  