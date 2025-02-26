import sys

import cv2
import numpy as np

import config
from landmarks import locateLandmarks
from utils import getFilenames, loadImages, showImage


def diminuirImagem(img):
  """Diminui a imagem para determinada quantidade de pixels (se ela for maior que essa quantidade)"""
  # x=((pixels)/(len(img[0])*len(img)))**0.5
  # if x<1 :
  #     print('asdf:', pixels, len(img[0]), len(img))
  #     img = cv2.resize(img,(int(len(img[0])*x),int(len(img)*x)), interpolation=cv2.INTER_AREA)
  # return img
  (height, width) = img.shape[:2]
  # print('size:',height,width)
  if height * width > 480000:
    heightFinal = 600
    widthFinal = (heightFinal * width) / height
    imgfinal = cv2.resize(img, (int(widthFinal), int(heightFinal)), interpolation=cv2.INTER_AREA)
    return imgfinal
  return img

def remove(img, Min):
  """Recebe uma imagem e um tamanho minimo, devolve a imagem apenas com os objetos maiores que o tamanho minimo"""
  nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
  sizes = stats[1:, -1] 
  nb_components = nb_components - 1
  min_size = Min
  img = np.zeros((output.shape),np.uint8)
  for i in range(0, nb_components):
    if sizes[i] >= min_size:
      img[output == i + 1] = 255
  return img

def find_nearest_white(img, target):
  """Da uma localizacao e devolve a localizacao do ponto branco mais proximo, sera usado para manter apenas a asa centralizada"""
  img = remove(img, 2000)
  nonzero2 = cv2.findNonZero(img)           
  distances = np.sqrt((nonzero2[:,:,0] - target[1]) ** 2 + (nonzero2[:,:,1] - target[0]) ** 2)
  nearest_index = np.argmin(distances)
  nearest_white= nonzero2[nearest_index]
  return nearest_white

def removeWings(img,kernel):
  """Metodo que deixa apenas a asa centralizada"""
  """Dropbox - > ferias -> Tirarruidosenormes2 explica o algoritmo""" 

  big = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

  target = int(len(big)/2) , int(len(big[0])/2)
  TARGET = find_nearest_white(big,target)
  nb_components2, output2, stats2, centroids2 = cv2.connectedComponentsWithStats(big, connectivity=8)
  nb_components2 = nb_components2 - 1
  imga = np.zeros((output2.shape),np.uint8)
  booleano = 0
  for i in range(0, nb_components2):
    if booleano == 0 :
      imgtransit = np.zeros((output2.shape),np.uint8)
      imgtransit[output2 == i + 1] = 255
      if imgtransit[TARGET[0][1]][TARGET[0][0]] == 255:
        imga[output2 == i + 1] = 255
        booleano == 1
  img_bwo2 = cv2.bitwise_and(imga, img)
  return img_bwo2

def cutImage(img):
  """Devolve o bounding box da asa binarizada xx,ww,hh,yy
  Alem disso, zcolsums e uma variavel que possui a soma dos pixels de cada coluna da matriz da imagem,
  e uma maneira de identificar se a asa esta inteira na imagem ou esta cortada, pois se a primeira coluna for = 0, significa que 
  nao existe pixels da asa nessa parte inicial da matriz.
  """
  zcolsums = np.sum(img, axis=0)
  zlines = np.sum(img, axis=1)
  zcolsums2 = np.nonzero(0-zcolsums)                                                                 
  zlines2 = np.nonzero(0-zlines)    
  xx=zlines2[0][0]                                                               
  yy=zlines2[0][zlines2[0].shape[0]-1]    
  ww=zcolsums2[0][0]
  hh=zcolsums2[0][zcolsums2[0].shape[0]-1]
  if xx > 7 :
    xx = xx-8
  else :
    xx = 0 
      
  if ww > 7 :
    ww = ww-8
  else :
    ww = 0
  
  if hh < img.shape[1] -9:
    hh=hh+8
  else :
    hh=img.shape[1]-1
      
  if yy < img.shape[0] -9:
    yy=yy+8
  else :
    yy=img.shape[0]-1
  return xx,ww,hh,yy

def differenceOfGassians(img):
  """Metodo de binarizacao chamado difference of gaussians"""
  g1 =  cv2.GaussianBlur(img,(19,19),0)
  g2=  cv2.GaussianBlur(img,(21,21),0)
  result = g1 - g2
  showImage('dog', result)
  _,img = cv2.threshold(result,128,255,cv2.THRESH_BINARY)
  return img

def remove(img, Min):
  """Recebe uma imagem e um tamanho minimo, devolve a imagem apenas com os objetos maiores que o tamanho minimo"""
  nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
  sizes = stats[1:, -1] 
  nb_components = nb_components - 1
  min_size = Min
  img = np.zeros((output.shape),np.uint8)
  for i in range(0, nb_components):
    if sizes[i] >= min_size:
      img[output == i + 1] = 255
  return img


def applyFilters(image):
  kernel9 = np.ones((3,3), np.uint8)

  # filteredImage = diminuirImagem(image)
  # originalImage = diminuirImagem(image)
  # utils.showImage('image original', originalImage)

  filteredImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  showImage('escala de cinza', filteredImage)
  filteredImage = cv2.medianBlur(filteredImage, 5)
  showImage('filtro de media', filteredImage)
  filteredImage = cv2.bilateralFilter(filteredImage, 5, 150, 150)
  showImage('filtro bilateral', filteredImage)
  filteredImage = differenceOfGassians(filteredImage)
  showImage('Thresshold', filteredImage)
  filteredImage = remove(filteredImage,90)
  showImage('remocao de ruidos', filteredImage)
  filteredImage = cv2.dilate(filteredImage, kernel9, iterations=1)
  showImage('dilatacao', filteredImage)
  filteredImage=cv2.erode(filteredImage,kernel9,iterations=1)
  showImage('erosao', filteredImage)
  # filteredImage=removeWings(filteredImage,kernel9)

  # xx,ww,hh,yy = cutImage(filteredImage) 
  # filteredImage = filteredImage[xx:yy, ww:hh]
  # originalImage = originalImage[xx:yy, ww:hh]

  filteredImage = cv2.ximgproc.thinning(filteredImage,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
  showImage('esqueletizacao', filteredImage)

  filteredImage = remove(filteredImage,150)
  showImage('remocao de ruidos', filteredImage)
  # filteredImage = removeStubs(filteredImage)
  return filteredImage

def saveLandmarks(image, filename):
  path = filename.replace(config.DATASET_ORIGINAL, config.LANDMARKS_PATH)
  cv2.imwrite(path, image)

def saveNpys(landmarks, filename):
  path = filename.replace(config.DATASET_ORIGINAL, config.NPY_PATH)
  path = path.replace('.png', '')
  np.save(path, landmarks)

def main():
  n = len(sys.argv)

  if n == 2:
    filenames = [ sys.argv[1] ]
  else:
    filenames = getFilenames(config.DATASET_ORIGINAL, config.DATASET_IN_EXTENSION)
  
  images = loadImages(filenames)

  h = []
  w = []
  for index in range(0, len(images)):
    print('Processing ', filenames[index])
    print(images[index].shape)
    height, width, c = images[index].shape
    h.append(height)
    w.append(width)
    
    filteredImage = applyFilters(images[index])
    filteredImage, landmarks = locateLandmarks(filteredImage)
    showImage('landmarks', filteredImage)
    saveLandmarks(filteredImage, filenames[index])
    saveNpys(landmarks, filenames[index])

  # print("mix height: ", min(h), " max height: ", max(h))
  # print("mix width: ", min(w), " max width: ", max(w))
main()