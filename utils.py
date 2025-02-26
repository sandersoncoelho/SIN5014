import glob

import cv2
from matplotlib import pyplot as plt

import config


def showImage(title, image):
  # plt.title(title)
  # plt.imshow(image)
  # plt.show()
  cv2.imshow(title, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def getFilenames(directory, extension):
  filenames = []
  _directory = glob.glob(directory)
  _extension = "/*." + extension

  for index in range(len(_directory)):
    if glob.glob(_directory[index] + _extension) != []:
      filenames.append(glob.glob(_directory[index] + _extension))

  return filenames[0]

def loadImages(filenames):
  images = []

  for index in range(len(filenames)):
    image = cv2.imread(filenames[index])
    images.append(image)

  return images

def saveImageOut(image, filename, filenamePrefix):
    path = filename.replace(config.DATASET_ORIGINAL, config.LANDMARKS_PATH + '/images/' + filenamePrefix)
    path = path.replace(config.DATASET_IN_EXTENSION, config.DATASET_OUT_EXTENSION)
    print(path)
    cv2.imwrite(path, image)