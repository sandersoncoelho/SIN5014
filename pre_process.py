import sys

from basic_filters import applyFilters
from landmarks import locateLandmarks, saveLandmarks
from utils import getFilenames, loadImages, saveImageOut


def main():
    filenames = getFilenames()
    images = loadImages(filenames)

    for index in range(0, len(images)):
        print('Processing ', filenames[index])
        
        image = applyFilters(images[index])
        image, landmarks = locateLandmarks(image)

        saveImageOut(image, filenames[index])
        saveLandmarks(landmarks, filenames[index])

main()

