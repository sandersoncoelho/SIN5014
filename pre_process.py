import sys

import config
from basic_filters import applyFilters
from landmarks import locateLandmarks, saveLandmarks
from utils import getFilenames, loadImages, saveImageOut, showImage


def main():
    n = len(sys.argv)

    if n == 2:
        filenames = [ sys.argv[1] ]
    else:
        filenames = getFilenames(config.DATASET_ORIGINAL, config.DATASET_IN_EXTENSION)
    
    images = loadImages(filenames)

    for index in range(0, len(images)):
        print('Processing ', filenames[index])
        
        filteredImage, originalImage = applyFilters(images[index])
        filteredImage, landmarks = locateLandmarks(filteredImage)
        # showImage('landmarks', filteredImage)

        saveImageOut(filteredImage, filenames[index], 'filtered')
        saveImageOut(originalImage, filenames[index], 'original')
        saveLandmarks(landmarks, filenames[index])
        # saveNpys(landmarks, filenames[index])

main()

