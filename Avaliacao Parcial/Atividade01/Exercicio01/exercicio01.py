import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/Atividade01/Exercicio01/forms.png" -O "forms.png"

img = cv.imread("forms.png")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

circle_color_contour = (255, 0, 0)
rectangle_color_contour = (0, 200, 0)
box_color_contour = (0, 0, 0)
thickness_contour = 3

ret, threshed_img = cv.threshold(img2,
                                 127, 
                                 255, 
                                 cv.THRESH_BINARY)

contours, order = cv.findContours(threshed_img, 
                                  cv.RETR_TREE, 
                                  cv.CHAIN_APPROX_SIMPLE)

for c in contours:
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(img, 
                 (x, y), 
                 (x+w, y+h), 
                 rectangle_color_contour, 
                 thickness_contour)

    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, 
                    [box], 
                    0, 
                    box_color_contour,
                    thickness_contour)

    (x, y), radius = cv.minEnclosingCircle(c)
    center = (int(x), 
              int(y))
    
    radius = int(radius)

    img = cv.circle(img, 
                    center, 
                    radius, 
                    circle_color_contour, 
                    thickness_contour)


plt.figure(figsize=(15,15))
plt.imshow(img)
