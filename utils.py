import glob

import cv2

import config


def showImage(title, image):
  cv2.imshow(title, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def getFilenames():
  filenames = []
  dataset = glob.glob(config.DATASET_PATH)
  extension = "/*." + config.DATASET_IN_EXTENSION

  for index in range(len(dataset)):
    if glob.glob(dataset[index] + extension) != []:
      filenames.append(glob.glob(dataset[index] + extension))

  return filenames[0]

def loadImages(filenames):
  images = []

  for index in range(len(filenames)):
    image = cv2.imread(filenames[index])
    images.append(image)

  return images

def saveImageOut(image, filename):
    path = filename.replace(config.DATASET_PATH, config.OUT_PATH + '/images')
    path = path.replace(config.DATASET_IN_EXTENSION, config.DATASET_OUT_EXTENSION)
    cv2.imwrite(path, image)