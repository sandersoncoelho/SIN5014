import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

import config
from landmarks import getLandmarksFromAnnotation, locateLandmarks
from utils import getFilenames

PROXIMITY_MEASURE = 10
MIN_LANDMARKS_ACCEPTED = 1

def getModifiedHausdorffDistance(img1, img2):    
  a=distance.cdist(img1,img2,'euclidean').min(axis=0)
  a= np.mean(a)

  b=distance.cdist(img2,img1,'euclidean').min(axis=0)
  b= np.mean(b)
  
  return round((a+b)/2,5)

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
  plotHausdorff(hausdorffDistances)
  plotExpectedsAndFoundout(expected, foundout)

main()
