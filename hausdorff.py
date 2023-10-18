import math

import cv2
import numpy as np
from scipy.spatial import distance

import config
from landmarks import getLandmarksFromAnnotation, locateLandmarks
from utils import getFilenames

COUNT_LANDMARKS = 10
POINTS_CLOSE_MEASURE = 10

def showImage():
  image1 = cv2.imread("./out_test/images/filtered/m6e haploide.png")
  image2 = cv2.imread("./out_test/images/original/m6e haploide.png")

  cv2.imshow('image1', image1)
  cv2.imshow('image2', image2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def getModifiedHausdorffDistance(img1, img2):    
  a=distance.cdist(img1,img2,'euclidean').min(axis=0)
  a= np.mean(a)

  b=distance.cdist(img2,img1,'euclidean').min(axis=0)
  b= np.mean(b)
  
  return round((a+b)/2,5)

def isPointsClose(center, point, radius):
  return pow(point[0] - center[0], 2) + pow(point[1] - center[1], 2) < pow(radius, 2)

def main():
  landmarksAnnotation = getLandmarksFromAnnotation('./annotation/via_project_14Oct2023_8h45m.json', COUNT_LANDMARKS)
  # print('annotation:\n', landmarksAnnotation)

  filenameNpys = getFilenames(config.OUT_PATH + '/landmarks', config.NPY_EXTENSION)
  for filenameNpy in filenameNpys:
    # if 'm6e haploide' in filenameNpy:
      landmarksNpy = np.load(filenameNpy, allow_pickle=True).tolist()
      # print('npy:\n', landmarksNpy)

  landmarkAnnClose = []
  landmarkNpyClose = []

  for landmarkAnn in landmarksAnnotation:
    for landmarkNpy in landmarksNpy:
      # print('annotation: ', landmarkAnn, ', npy: ', landmarkNpy)
      if isPointsClose(landmarkAnn, landmarkNpy, POINTS_CLOSE_MEASURE) and \
        landmarkAnn not in landmarkAnnClose and \
          landmarkNpy not in landmarkNpyClose:
            landmarkAnnClose.append(landmarkAnn)
            landmarkNpyClose.append(landmarkNpy)


  
  print('landmarkAnnClose:\n', landmarkAnnClose)
  print('landmarkNpyClose:\n', landmarkNpyClose)
  print('distance modified hausdorff\n', getModifiedHausdorffDistance(landmarkAnnClose, landmarkNpyClose))

  image = cv2.imread("./out_test/images/filtered/m6e haploide.png")
  for p in landmarkNpyClose:
    x = p[0]
    y = p[1]
    image = cv2.circle(image, (x, y), 5, 100, -1)
  cv2.imshow('image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  # showImage()
  # print('distance modified hausdorff\n', getModifiedHausdorffDistance([[0, 1]], [[-0.5, 0.75]]))

main()

# p = [452,  13]
# q = [461, 14]
# print('distance:', math.dist(p, q))
# print('teste: ', isPointsClose(p, q, 30))

# p = [408, 15]
# q = [400, 19]
# print('distance:', math.dist(p, q))
# print('teste: ', isPointsClose(p, q, 30))