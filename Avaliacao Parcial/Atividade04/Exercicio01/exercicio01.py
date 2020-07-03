# Exercicio 01
# Pedro Afonso Ferreira Haupenthal 823974

# OBS
# Imagem celular original do vírus recentemente descoberto SARS-CoV-2, 
# popularmente chamado de COVID-19 ou Coronavirus.

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/Atividade04/Exercicio01/covid_19.jpg" -O "covid_19.jpg"

img1 = cv.imread("covid_19.jpg")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

img2 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2 = cv.bitwise_not(img2)
ret, img2 = cv.threshold(img2,
                         0,
                         255,
                         cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)

img3 = cv.morphologyEx(img2, 
                       cv.MORPH_CLOSE, 
                       kernel, 
                       iterations = 2)
img4 = cv.dilate(img3,
                 kernel,
                 iterations = 5)

img5 = cv.distanceTransform(img3, 
                            cv.DIST_L2, 
                            5)
ret,img6 = cv.threshold(img5,
                        0.65 * img5.max(), 
                        255, 
                        0)

img6 = np.uint8(img6)
img7 = cv.subtract(img4, img6)

ret, count = cv.connectedComponents(img6)
count = count + 1

count[img7 == 255] = 0
img8 = cv.watershed(img1, count)
img1[count == -1] = [255, 0, 0]


plt.figure(figsize=(30,30))
plt.subplot(121), plt.imshow(img1), plt.title("ORIGINAL"), plt.axis("off")
plt.subplot(122), plt.imshow(img8, cmap='jet'), plt.title("RESULTADO"), plt.axis("off")
plt.show()