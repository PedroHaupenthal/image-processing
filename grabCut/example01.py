import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/grabCut/bmw.jpg" -O "bmw.jpg"

img1 = cv.imread("bmw.jpg")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

img2 = img1.copy()
p1 = (100, 194)
p2 = (597, 290)
img2 = cv.rectangle(img2, p1, p2, (255,0,0), 2)

mask = np.zeros(img1.shape[:2], np.uint8)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)
rectangle = p1 + p2

cv.grabCut(img1, 
           mask, 
           rectangle, 
           bgModel, 
           fgModel, 
           5, 
           cv.GC_INIT_WITH_RECT)

filter = np.where ( (mask == 0) | (mask == 2), 0, 1).astype('uint8')
img3 = img1.copy()
img3 = img3 * filter[:, :, np.newaxis]

plt.figure(figsize=(25,25))
plt.subplot(131), plt.imshow(img1)
plt.subplot(132), plt.imshow(img2)
plt.subplot(133), plt.imshow(img3)