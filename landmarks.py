import json
import math
import os

import cv2
import numpy as np

import config
from utils import getFilenames


def getLandmarks(input_image):
  """Metodo que recebe uma imagem esquelitzada e devolve uma lista contendo os pontos de interesse"""
  
  """Templetes de interesse"""
  kernel = np.array((
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 0]), dtype="int")

  kernel2 = np.array((
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0]), dtype="int")
  
  kernel3 = np.array((
    [0, 0, 1],
    [1, 1, 0],
    [0, 0, 1]), dtype="int")
  
  kernel4 = np.array((
    [1, 0, 0],
    [0, 1, 1],
    [0, 1, 0]), dtype="int")
  
  kernel5 = np.array((
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 1]), dtype="int")
  
  kernel6 = np.array((
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0]), dtype="int")
  
  kernel7 = np.array((
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 0]), dtype="int")
  
  kernel8 = np.array((
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]), dtype="int")
  
  kernel9 = np.array((
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 1]), dtype="int")
  
  kernel10 = np.array((
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0]), dtype="int")
  
  kernel11 = np.array((
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 1]), dtype="int")
  
  kernel12 = np.array((
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1]), dtype="int")
  
  """Unindo todos pontos de interesse"""
  output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
  output_image2 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel2)
  output_image3 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel3)
  output_image4 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel4)
  output_image5 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel5)
  output_image6 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel6)
  output_image7 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel7)
  output_image8 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel8)
  output_image9 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel9)
  output_image10 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel10)
  output_image11 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel11)
  output_image12 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel12)
  output_image = cv2.bitwise_or(output_image,output_image2)
  output_image2 = cv2.bitwise_or(output_image3,output_image4)
  output_image3 = cv2.bitwise_or(output_image5,output_image6)
  output_image4 = cv2.bitwise_or(output_image7,output_image8)
  output_image5 = cv2.bitwise_or(output_image9,output_image10)
  output_image6 = cv2.bitwise_or(output_image11,output_image12)
  output_image8 = cv2.bitwise_or(output_image,output_image2)
  output_image9 = cv2.bitwise_or(output_image3,output_image4)
  output_image10 = cv2.bitwise_or(output_image5,output_image6)
  output_image11 = cv2.bitwise_or(output_image8,output_image9)
  output_image12 = cv2.bitwise_or(output_image10,output_image11)
  i,j = np.where(output_image12 == 255)
  intersections=[]
  
  """Caso a asa esteja inteira na imagem, excluir os pontos de interesse perto das bordas laterais"""
  zcolsums = np.sum(input_image, axis=0)
  for x in range(i.shape[0]):

    if zcolsums[0]==0 and zcolsums[zcolsums.shape[0]-1]==0:
      lateral1=int(output_image12.shape[1]/5)
      lateral2=int(output_image12.shape[1]/8)

    else :
      lateral1=int(output_image12.shape[1]/25)
      lateral2=int(output_image12.shape[1]/25)

        
    if j[x] > lateral1 and i[x]< output_image12.shape[0]-2  and i[x]>1 and j[x]< output_image12.shape[1]-lateral2:
      intersections.append((j[x],i[x]))

  """Se tiver dois pontos de interesse muito pertos, deixar apenas um"""

  return intersections

def locateLandmarks(image):
  landmarks = getLandmarks(image)

  for p in landmarks:
    x = p[0]
    y = p[1]
    image = cv2.circle(image, (x, y), 5, 255, -1)
  
  return image, landmarks

def saveLandmarks(landmarks, filename):
  path = filename.replace(config.DATASET_PATH, config.OUT_PATH + '/landmarks')
  path = path.replace('.' + config.DATASET_IN_EXTENSION, '')
  np.save(path, landmarks)

def getLandmarksFromAnnotation(annotationFile):
  annotations = json.load(open(os.path.join('./', annotationFile)))
  image_metadata = annotations['_via_img_metadata']
  image_id_list =  annotations['_via_image_id_list']
  image_id_list.sort()

  allLandmarks = []
  for keyFilename in image_id_list:
    regions = image_metadata[keyFilename]['regions']

    instanceLandmarks = []
    for i in range(len(regions)):
      point = regions[i]['shape_attributes']
      p = [point['cx'], point['cy']]
      instanceLandmarks.append(p)

    allLandmarks.append(instanceLandmarks)

  return allLandmarks