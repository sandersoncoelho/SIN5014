
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

import config
from landmarks import getLandmarksFromAnnotation, mergeLandmarksClose
from utils import getFilenames, showImage

LANDMARKS = 18
PROXIMITY_MEASURE = 10


def getModifiedHausdorffDistance(arrayA, arrayB):    
  a = distance.cdist(arrayA, arrayB, 'euclidean').min(axis = 0)
  a = np.mean(a)

  b = distance.cdist(arrayB, arrayA, 'euclidean').min(axis = 0)
  b = np.mean(b)
  
  return max(a, b)

def isPointsClose(center, point, radius):
  return pow(point[0] - center[0], 2) + pow(point[1] - center[1], 2) < pow(radius, 2)

def saveImageMerged(filenameNpy, landmarkNpyClose):
  filename = filenameNpy.replace('/npy', '/original')
  filename = filename.replace('.npy', '.png')

  image = cv2.imread(filename)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  for p in landmarkNpyClose:
    x = p[0]
    y = p[1]
    image = cv2.circle(image, (x, y), 5, 255, -1)
  filename = filename.replace('/original', '/mhd')
  cv2.imwrite(filename, image)

def plotHausdorff(hausdorffDistances):
  plt.title("Distância Modificada de Hausdorff", fontsize=14)
  plt.xlabel("Instâncias de imagens", fontsize=12)
  plt.ylabel("Pixels", fontsize=12)
  plt.scatter(range(len(hausdorffDistances)), hausdorffDistances)
  plt.show()

  plt.title("Distância Modificada de Hausdorff", fontsize=14)
  plt.xlabel("Pixels", fontsize=12)
  plt.ylabel("Instâncias de imagens", fontsize=12)
  plt.hist(hausdorffDistances)
  plt.show()

def plotExpectedsAndFoundout(expected, foundout):
  expectedMean = np.mean(expected)
  foundoutMean = np.mean(foundout)
  
  plt.hlines(y=expectedMean, xmin=0, xmax=len(expected), linewidth=2, color='#07DE07')
  plt.hlines(y=foundoutMean, xmin=0, xmax=len(foundout), linewidth=2, color='#F15126')

  formatPrecision = lambda x : "%.1f" % x
  plt.text(len(expected), expectedMean, formatPrecision(expectedMean))
  plt.text(len(foundout), foundoutMean, formatPrecision(foundoutMean))

  plt.plot(expected, color='tab:green', label='Esperado')
  plt.plot(foundout, color='tab:red', label='Encontrado')
  # plt.scatter(range(len(expected)), expected)
  # plt.scatter(range(len(foundout)), foundout)
  
  plt.title("Landmarks encontrados x esperados", fontsize=14)
  plt.xlabel("Instâncias de imagens", fontsize=12)
  plt.ylabel("Landmarks", fontsize=12)
  plt.legend(fontsize=12)
  plt.tight_layout()
  plt.show()

def calculateMetrics(landmarksAnn, landmarksNpy):
  corrects = 0
  for landmarkAnn in landmarksAnn:
    for landmarkNpy in landmarksNpy:
      if isPointsClose(landmarkAnn, landmarkNpy, PROXIMITY_MEASURE):
        corrects += 1

  corrects = LANDMARKS if corrects > LANDMARKS else corrects
  errors = len(landmarksNpy) - corrects
  return corrects / LANDMARKS, errors / len(landmarksNpy)

def main():
  hausdorffDistances = []

  allLandmarksAnn = getLandmarksFromAnnotation('./annotation/AT_annotation.json')
  filenameNpys = getFilenames(config.NPY_PATH, config.NPY_EXTENSION)
  filenameNpys.sort()
  allLandmarksNpy = []
  allAccuracy = []
  allErrors = []

  for filenameNpy in filenameNpys:
    allLandmarksNpy.append(np.load(filenameNpy, allow_pickle=True).tolist())

  for i in range(len(allLandmarksAnn)):
    landmarksAnn = allLandmarksAnn[i]
    landmarksNpy = allLandmarksNpy[i]

    mhd = getModifiedHausdorffDistance(landmarksAnn, landmarksNpy)
    accuracy, errors = calculateMetrics(landmarksAnn, landmarksNpy)

    print('distance hausdorff modified: ', mhd)
    hausdorffDistances.append(mhd)
    allAccuracy.append(accuracy)
    allErrors.append(errors)

    saveImageMerged(filenameNpys[i], landmarksNpy)

  print('allAccuracy: ', allAccuracy)
  print('allErrors: ', allErrors)
  print('\n\nQuantidades de images disponíveis: ', len(allLandmarksAnn))
  print('MHD média:', np.mean(hausdorffDistances))
  print('Acurácia média: ', np.mean(allAccuracy))
  print('Erro médio: ', np.mean(allErrors))
  plotHausdorff(hausdorffDistances)
  # plotExpectedsAndFoundout(expected, foundout)

main()
import json
import os


def getLandmarksAnn():
  annotations = json.load(open(os.path.join('./', './annotation/dataset_out.json')))
  image_metadata = annotations['_via_img_metadata']

  regions = image_metadata['m1d haploide.png347955']['regions']

  instanceLandmarks = []
  for i in range(len(regions)):
    point = regions[i]['shape_attributes']
    p = [point['cx'], point['cy']]
    instanceLandmarks.append(p)

  return instanceLandmarks




def teste():
  landmarksAnn = np.array(getLandmarksAnn())
  print("landmarksAnn: ", landmarksAnn)

  filenameNpy = "./out_test/landmarks/m1d haploide.npy"
  landmarksNpy = np.load(filenameNpy, allow_pickle=True).tolist()
  print("landmarksNpy: ", landmarksNpy)
  landmarksNpy = mergeLandmarksClose(landmarksNpy)

  height, width = 254, 744
  image = np.zeros(height * width * 3).reshape(height, width, 3)

  for landmark in landmarksAnn:
    image = cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)

  firstColumn = landmarksAnn[:, 0]
  xMin, xMax = min(firstColumn), max(firstColumn)
  landmarksNpyResult = []
  for landmark in landmarksNpy:
    if xMin <= landmark[0] and landmark[0] <= xMax:
      image = cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255), -1)
      landmarksNpyResult.append(landmark)

  print("MHD:", getModifiedHausdorffDistance(landmarksAnn, landmarksNpyResult))

  showImage('teste', image)

# teste()
