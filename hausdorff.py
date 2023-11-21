import itertools
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.spatial import distance

import config
from landmarks import getLandmarksFromAnnotation, locateLandmarks
from utils import getFilenames, showImage

PROXIMITY_MEASURE = 22
MIN_LANDMARKS_ACCEPTED = 8
MERGE_MEASURE = 10

def getModifiedHausdorffDistance(arrayA, arrayB):    
  a = distance.cdist(arrayA, arrayB, 'euclidean').min(axis = 0)
  a = np.mean(a)

  b = distance.cdist(arrayB, arrayA, 'euclidean').min(axis = 0)
  b = np.mean(b)
  
  return max(a, b)

def mergeLandmarksClose(landmarksNpy):
  result = []
  pointsToBeRemoved = []

  combinations = list(itertools.combinations(landmarksNpy, 2))

  for pointA, pointB in combinations:
    if math.dist(pointA, pointB) < MERGE_MEASURE:
      mergedPoint = [int((pointA[0] + pointB[0])/2), int((pointA[1] + pointB[1]) / 2)]
      result.append(mergedPoint)
      pointsToBeRemoved.append(pointA)
      pointsToBeRemoved.append(pointB)

  for point in landmarksNpy:
    if point not in pointsToBeRemoved:
      result.append(point)

  return result

def isPointsClose(center, point, radius):
  return pow(point[0] - center[0], 2) + pow(point[1] - center[1], 2) < pow(radius, 2)

def saveImageMerged(filenameNpy, landmarkNpyClose):
  filename = filenameNpy.replace('/landmarks', '/images/filtered')
  filename = filename.replace('.npy', '.png')

  image = cv2.imread(filename)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  for p in landmarkNpyClose:
    x = p[0]
    y = p[1]
    image = cv2.circle(image, (x, y), 5, 100, -1)
  filename = filename.replace('/filtered', '/merged')
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

def main():
  hausdorffDistances = []
  expected = []
  foundout = []

  allLandmarksAnn = getLandmarksFromAnnotation('./annotation/dataset_out.json')
  filenameNpys = getFilenames(config.OUT_PATH + '/landmarks', config.NPY_EXTENSION)
  filenameNpys.sort()
  allLandmarksNpy = []
  for filenameNpy in filenameNpys:
    allLandmarksNpy.append(np.load(filenameNpy, allow_pickle=True).tolist())

  for i in range(len(allLandmarksAnn)):
    landmarksAnn = allLandmarksAnn[i]
    landmarksNpy = allLandmarksNpy[i]
    
    landmarkAnnClose = []
    landmarkNpyClose = []

    for landmarkAnn in landmarksAnn:
      for landmarkNpy in landmarksNpy:
        if isPointsClose(landmarkAnn, landmarkNpy, PROXIMITY_MEASURE) and \
          landmarkAnn not in landmarkAnnClose and \
            landmarkNpy not in landmarkNpyClose:
              landmarkAnnClose.append(landmarkAnn)
              landmarkNpyClose.append(landmarkNpy)

    if len(landmarkAnnClose) >= MIN_LANDMARKS_ACCEPTED:
      expected.append(len(landmarksAnn))
      foundout.append(len(landmarkAnnClose))
      print('\n\nlandmarkAnnClose:\n', landmarkAnnClose)
      print('landmarkNpyClose:\n', landmarkNpyClose)
      print('distance modified hausdorff\n', getModifiedHausdorffDistance(landmarkAnnClose, landmarkNpyClose))

      hausdorffDistances.append(getModifiedHausdorffDistance(landmarkAnnClose, landmarkNpyClose))
      saveImageMerged(filenameNpys[i], landmarkNpyClose)

  print('\n\nQuantidades de images disponíveis: ', len(allLandmarksAnn))
  print('Quantidades de images aceitas para o cálculo de mdh: ', len(hausdorffDistances))
  print('Limite mínino de landmarks aceito por imagem: ', MIN_LANDMARKS_ACCEPTED)
  print('Taxa de aproveitamento das imagens: ', len(hausdorffDistances) / len(allLandmarksAnn))
  print('Acurácia média: ', np.mean(foundout) / np.mean(expected))
  print('MDH média:', np.mean(hausdorffDistances))
  plotHausdorff(hausdorffDistances)
  plotExpectedsAndFoundout(expected, foundout)

# main()
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

  print("MDH:", getModifiedHausdorffDistance(landmarksAnn, landmarksNpyResult))

  showImage('teste', image)

teste()
