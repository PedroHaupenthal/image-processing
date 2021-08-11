# OBS
# A foto utilizada para a aplicacao do algoritmo foi tirada dia 12/04/2020, onde passou pelo processo
# de redução da resolução de 5183x3456 para 1280x853, visando um menor tempo de processamento.

# Apos a aplicao do grabCut inicialmente com um retangulo, foi utilizado o mesmo grabcut porem com um filtro
# como indica a documentação abaixo do OpenCV
# https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/grabCut/image.jpeg" -O "image.jpeg"

img1 = cv.imread("image.jpeg")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

img2 = img1.copy()
p1 = (80, 20)
p2 = (900, 820)
img2 = cv.rectangle(img2, p1, p2,(255,0,0),5)

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

filtro = np.where ((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img3 = img1.copy()
img3 = img3 * filtro[:, :, np.newaxis]

!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/grabCut/image-filter.jpeg" -O "image-filter.jpeg"

img_mask = cv.imread("image-filter.jpeg", 0)
mask[img_mask == 0] = 0
mask[img_mask == 255] = 1
mask, bgModel, fgModel = cv.grabCut(img1,
                                    mask,
                                    None,
                                    bgModel,
                                    fgModel,
                                    5,
                                    cv.GC_INIT_WITH_MASK)

mask = np.where ((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img4 = img1*mask[:, :, np.newaxis]
img_mask = cv.cvtColor(img_mask, cv.COLOR_BGR2RGB)

plt.figure(figsize=(100,100))
plt.subplot(331), plt.imshow(img1), plt.axis("off")
plt.subplot(332), plt.imshow(img2), plt.axis("off")
plt.subplot(333), plt.imshow(img3), plt.axis("off")
plt.subplot(334), plt.imshow(img_mask), plt.axis("off")
plt.subplot(335), plt.imshow(img4), plt.axis("off")