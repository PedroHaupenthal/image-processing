# Exercicio 03
# Pedro Afonso Ferreira Haupenthal 823974

# OBS
# 1º O algoritmo so pode ser executado no Google Colab devido a importações específicas
# 2º Video capturado com a camera frontal do celular e utilizado o software online clideo.com para redimensionar, passando de 1980x1080 para 600x600
# e ajustar a proporção desejada de 18:9 para 1:1

from google.colab.patches import cv2_imshow as colab_imshow
import numpy as np
import cv2 as cv

!wget "https://github.com/PedroHaupenthal/Image-Processing/blob/master/Atividade05/base_files/eu.mp4?raw=true" -O "eu.mp4"
!wget "https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/Atividade05/base_files/cascades/haarcascade_frontalface_default.xml" -O "haarcascade_frontalface_default.xml"

video = cv.VideoCapture("eu.mp4")
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    _, frame = video.read()

    img1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img1, 
                                          1.3, 
                                          5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, 
                      (x ,y),
                      (x + w, y + h),
                      (255, 200, 0), 10)
        roi_im1 = img1[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    colab_imshow(frame)
