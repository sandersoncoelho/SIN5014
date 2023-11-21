import itertools
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import config
from landmarks import getLandmarksFromAnnotation
from mathUtils import getAngle
from polygon import (AngleMeanStd, AngleStandart, EdgeMeanStd, EdgeStandart,
                     Polygon)
from utils import getFilenames, showImage

EDGE_PROXIMITY_MEASURE = 15
MERGE_MEASURE = 10

def isSimilarPolygons(polygonA, polygonB, edgeError, angleError):
  edgeSimilarities = [
    polygonA.edge12 / polygonB.edge12,
    polygonA.edge24 / polygonB.edge24,
    polygonA.edge45 / polygonB.edge45,
    polygonA.edge57 / polygonB.edge57,
    polygonA.edge79 / polygonB.edge79,
    polygonA.edge910 / polygonB.edge910,
    polygonA.edge108 / polygonB.edge108,
    polygonA.edge86 / polygonB.edge86,
    polygonA.edge63 / polygonB.edge63,
    polygonA.edge31 / polygonB.edge31
  ]
  # print("edge std: ", np.std(edgeSimilarities))
  if np.std(edgeSimilarities) > edgeError: return False

  angleSimilarities = [
    polygonA.angle1 / polygonB.angle1,
    polygonA.angle2 / polygonB.angle2,
    polygonA.angle3 / polygonB.angle3,
    polygonA.angle4 / polygonB.angle4,
    polygonA.angle5 / polygonB.angle5,
    polygonA.angle6 / polygonB.angle6,
    polygonA.angle7 / polygonB.angle7,
    polygonA.angle8 / polygonB.angle8,
    polygonA.angle9 / polygonB.angle9,
    polygonA.angle10 /polygonB.angle10 
  ]
  # print("angle std: ", np.std(angleSimilarities))
  if np.std(angleSimilarities) > angleError: return False

  return True

def getLandmarksAnnotationDataset():
  allLandmarksAnnotation = getLandmarksFromAnnotation('./annotation/dataset_out.json')
  allLandmarksAnnotationValid = []
  edge12s, angle1s = [], []
  edge24s, angle2s = [], []
  edge45s, angle3s = [], []
  edge57s, angle4s = [], []
  edge79s, angle5s = [], []
  edge910s, angle6s = [], []
  edge108s, angle7s = [], []
  edge86s, angle8s = [], []
  edge63s, angle9s = [], []
  edge31s, angle10s = [], []

  for i in range(len(allLandmarksAnnotation)):
    landmarksAnnotation = allLandmarksAnnotation[i]

    if len(landmarksAnnotation) == 10:
      polygon = Polygon(*landmarksAnnotation)
      allLandmarksAnnotationValid.append(polygon)

      edge12s.append(polygon.edge12); angle1s.append(polygon.angle1)
      edge24s.append(polygon.edge24); angle2s.append(polygon.angle2)
      edge45s.append(polygon.edge45); angle3s.append(polygon.angle3)
      edge57s.append(polygon.edge57); angle4s.append(polygon.angle4)
      edge79s.append(polygon.edge79); angle5s.append(polygon.angle5)
      edge910s.append(polygon.edge910); angle6s.append(polygon.angle6)
      edge108s.append(polygon.edge108); angle7s.append(polygon.angle7)
      edge86s.append(polygon.edge86); angle8s.append(polygon.angle8)
      edge63s.append(polygon.edge63); angle9s.append(polygon.angle9)
      edge31s.append(polygon.edge31); angle10s.append(polygon.angle10)

  edgeStandart = EdgeStandart(
    EdgeMeanStd(np.mean(edge12s), np.std(edge12s)),
    EdgeMeanStd(np.mean(edge24s), np.std(edge24s)),
    EdgeMeanStd(np.mean(edge45s), np.std(edge45s)),
    EdgeMeanStd(np.mean(edge57s), np.std(edge57s)),
    EdgeMeanStd(np.mean(edge79s), np.std(edge79s)),
    EdgeMeanStd(np.mean(edge910s), np.std(edge910s)),
    EdgeMeanStd(np.mean(edge108s), np.std(edge108s)),
    EdgeMeanStd(np.mean(edge86s), np.std(edge86s)),
    EdgeMeanStd(np.mean(edge63s), np.std(edge63s)),
    EdgeMeanStd(np.mean(edge31s), np.std(edge31s))
  )

  angleStandart = AngleStandart(
    AngleMeanStd(np.mean(angle1s), np.std(angle1s)),
    AngleMeanStd(np.mean(angle2s), np.std(angle2s)),
    AngleMeanStd(np.mean(angle3s), np.std(angle3s)),
    AngleMeanStd(np.mean(angle4s), np.std(angle4s)),
    AngleMeanStd(np.mean(angle5s), np.std(angle5s)),
    AngleMeanStd(np.mean(angle6s), np.std(angle6s)),
    AngleMeanStd(np.mean(angle7s), np.std(angle7s)),
    AngleMeanStd(np.mean(angle8s), np.std(angle8s)),
    AngleMeanStd(np.mean(angle9s), np.std(angle9s)),
    AngleMeanStd(np.mean(angle10s), np.std(angle10s))
  )

  return allLandmarksAnnotationValid, edgeStandart, angleStandart

# def testIsSimilarPolygons(allLandmarksAnnotationValid):
#   combinations = list(itertools.combinations(allLandmarksAnnotationValid, 2))

#   for polygonA, polygonB in combinations:
#     polygonsSimilar = isSimilarPolygons(polygonA, polygonB, 0.3, 0.3)
#     if not polygonsSimilar:
#       print("\npolygonA: ", polygonA)
#       print("polygonB: ", polygonB)
      # print("isSimilarPolygons: ", )

def getAllLandmarksNpy():
  filenameNpys = getFilenames(config.OUT_PATH + '/landmarks', config.NPY_EXTENSION)
  filenameNpys.sort()
  allLandmarksNpy = []
  print(filenameNpys[0])
  for filenameNpy in filenameNpys:
    allLandmarksNpy.append(np.load(filenameNpy, allow_pickle=True).tolist())

  return allLandmarksNpy

def getAllLandmarksNpyValid(allLandmarksNpy):
  allLandmarksNpyValid = []
  for landmarksNpy in allLandmarksNpy:
    lenght = len(landmarksNpy)
    
    if lenght >= 10:
      allLandmarksNpyValid.append(landmarksNpy)

  return allLandmarksNpyValid

def mergeLandmarksClose(landmarksNpy):
  result = []
  pointsToBeRemoved = []

  combinations = list(itertools.combinations(landmarksNpy, 2))

  for pointA, pointB in combinations:
    if math.dist(pointA, pointB) < MERGE_MEASURE:
      mergedPoint = [int((pointA[0] + pointB[0])/2), int((pointA[1] + pointB[1]) / 2)]
      print("mergedPoint: ", mergedPoint, " pointA:", pointA, " pointB: ", pointB)
      result.append(mergedPoint)
      pointsToBeRemoved.append(pointA)
      pointsToBeRemoved.append(pointB)

  for point in landmarksNpy:
    if point not in pointsToBeRemoved:
      result.append(point)

  return result

def isEdgeCandidate(distance, edgeMeanStd):
  return edgeMeanStd.edgeMean - EDGE_PROXIMITY_MEASURE <= distance and \
    distance <= edgeMeanStd.edgeMean + EDGE_PROXIMITY_MEASURE

def isAngleCandidate(pointA, pointB, pointC, angleMeanStd):
  angle = getAngle(pointB, pointA, pointC)
  if angleMeanStd.angleMean - angleMeanStd.angleStd <= angle and \
      angle <= angleMeanStd.angleMean + angleMeanStd.angleStd:
    return True
  
  angle = getAngle(pointA, pointB, pointC)
  if angleMeanStd.angleMean - angleMeanStd.angleStd <= angle and \
      angle <= angleMeanStd.angleMean + angleMeanStd.angleStd:
    return True
  
  angle = getAngle(pointA, pointC, pointB)
  if angleMeanStd.angleMean - angleMeanStd.angleStd <= angle and \
      angle <= angleMeanStd.angleMean + angleMeanStd.angleStd:
    return True
  
  return False

def getLeftRightPoints(pointA, pointB):
  if pointA[1] < pointB[1]: return pointA, pointB
  return pointB, pointA

def getLandmarksCandidate(landmarksNpy, edgeStandart, angleStandart):
  combinations = list(itertools.combinations(landmarksNpy, 2))

  p1Candidates = []
  p2Candidates = []
  
  image = np.zeros(275 * 752).reshape(275, 752)

  for pointA, pointB in combinations:
    distance = math.dist(pointA, pointB)

    if isEdgeCandidate(distance, edgeStandart.edge12):
      p1, p2 = getLeftRightPoints(pointA, pointB)
      p1Candidates.append(p1)
      p2Candidates.append(p2)
      image = cv2.circle(image, p1, 5, 100, -1)
      image = cv2.circle(image, p2, 5, 100, -1)
      image = cv2.line(image, p1, p2, 1)
      
  showImage("teste", image)
  print("p1Candidates:", p1Candidates)
  print("p2Candidates:", p2Candidates)

def plotLandmarks(landmarksNpy):
  image = np.zeros(275 * 752).reshape(275, 752)
  for landmark in landmarksNpy:
    image = cv2.circle(image, (landmark[0], landmark[1]), 5, 100, -1)
  showImage("teste", image)

def mean_confidence_interval(data, confidence=0.97):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def main():
  allLandmarksAnnotationValid, edgeStandart, angleStandart = getLandmarksAnnotationDataset()
  # annotations = []
  # for landmarksAnnotation in allLandmarksAnnotationValid:
  #   annotations.append(landmarksAnnotation.edge12)
  # # print(y)
  # # print(min(y), max(y))
  # # print(np.mean(y))
  # # print(np.std(y))
  # plt.hist(annotations)
  
  # print("sdt: ", edgeStandart.edge12.edgeMean, edgeStandart.edge12.edgeStd)

  allLandmarksNpy = getAllLandmarksNpy()
  allLandmarksNpy = getAllLandmarksNpyValid(allLandmarksNpy)
  landmarksNpy = allLandmarksNpy[0]
  landmarksNpy = mergeLandmarksClose(landmarksNpy)
  landmarksNpy = mergeLandmarksClose(landmarksNpy)
  
  # print(landmarksNpy)
  getLandmarksCandidate(landmarksNpy, edgeStandart, angleStandart)

  # plt.show()


main()
