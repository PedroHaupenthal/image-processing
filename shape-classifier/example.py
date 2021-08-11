
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Constants
MIN_IMAGE_NUMBER = 1
MAX_IMAGE_NUMBER = 300
FILE_FORMAT = '.bmp'

# Objects
class image_obj:
  def __init__(self, number, diameter, perimeter, anatomy, 
               area, compacity, momentum, solidity):
    self.number = number
    self.diameter = diameter
    self.perimeter = perimeter
    self.anatomy = anatomy
    self.area = area
    self.compacity = compacity
    self.momentum = momentum
    self.solidity = solidity

# Variables
images_list = []
images_list.append(["#","DIAMETRO","PERIMETRO","ESQUELETO","AREA",
                    "COMPACIDADE","MOMENTO","SOLIDEZ"])

for i in range(MIN_IMAGE_NUMBER, MAX_IMAGE_NUMBER + 1):
  image = image_obj

  img1 = cv.imread(f'{i}{FILE_FORMAT}')
  print("Processando imagem {0}{1} ...".format(i, FILE_FORMAT))

  img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

  img2 = img1.copy()
  img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
  _,img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)

  contorno, ordem = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  diametro = np.sqrt(4 * cv.contourArea(contorno[0]) / np.pi)

  perimetro = cv.countNonZero(cv.Canny(img2, 50, 100))

  img3 = np.zeros(img2.shape, np.uint8)
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

  end = False
  img4 = img2.copy()
  while (not end):
    erosao = cv.erode(img4, kernel)
    dilatacao = cv.dilate(erosao, kernel)
    subt = cv.subtract(img4, dilatacao)
    img3 = cv.bitwise_or(img3, subt)
    img4 = erosao.copy()

    zeros = np.size(img4) - cv.countNonZero(img4)
    if (zeros == np.size(img4)):
      end = True

  esqueleto = cv.countNonZero(img3)

  area = cv.countNonZero(img2)

  compacidade = np.square(perimetro) / area

  momentos = cv.moments(contorno[0])
  momento = int(momentos['m10']/momentos['m00'])

  area_obj = cv.contourArea(contorno[0])
  area_convex = cv.contourArea( cv.convexHull(contorno[0]))
  solidez = area_obj / area_convex

  image.number = i
  image.diameter = round(diametro, 2)
  image.perimeter = round(perimetro, 2)
  image.anatomy = round(esqueleto, 2)
  image.area = round(area, 2)
  image.compacity = round(compacidade, 2)
  image.momentum = round(momento, 2)
  image.solidity = round(solidez, 2)

  images_list.append([image.number,
                      image.diameter,
                      image.perimeter,
                      image.anatomy,
                      image.area,
                      image.compacity,
                      image.momentum,
                      image.solidity,
                    ])

np.savetxt("result.csv", images_list, delimiter=";", fmt="%s")
print("\nAnalise finalizada e salva no arquivo result.csv")
