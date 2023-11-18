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
  print('MDH média:', np.mean(hausdorffDistances))
  plotHausdorff(hausdorffDistances)
  plotExpectedsAndFoundout(expected, foundout)

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

import numpy as np


def ModHausdorffDist(A,B):
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)


def teste():
  landmarksAnn = getLandmarksAnn()
  print("landmarksAnn: ", landmarksAnn)

  filenameNpy = "./out_test/landmarks/m1d haploide.npy"
  landmarksNpy = np.load(filenameNpy, allow_pickle=True).tolist()
  print("landmarksNpy: ", landmarksNpy)

  height, width = 254, 744
  image = np.zeros(height * width * 3).reshape(height, width, 3)
  # image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

  for landmark in landmarksAnn:
    image = cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)

  for landmark in landmarksNpy:
    image = cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255), -1)

  print("MDH:", getModifiedHausdorffDistance(landmarksAnn, landmarksNpy))

  # set1 = np.array([[1, 2], [3, 4], [5, 6]])
  # set2 = np.array([[2, 3], [4, 5], [6, 3]])
  distance = ModHausdorffDist(np.array(landmarksAnn), np.array(landmarksNpy))
  # distance = modified_hausdorff_distance(np.array(landmarksAnn), np.array(landmarksNpy)))
  print(f"A distância modificada de Hausdorff entre os conjuntos é: {distance}")

  showImage('teste', image)

# teste()
