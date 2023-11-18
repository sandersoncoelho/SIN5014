import cv2
import numpy as np
from skimage import exposure

import utils
from basic_filters import differenceOfGassians, diminuirImagem, remove

# Equalization
# img_eq = exposure.equalize_hist(img)
# utils.showImage('img_eq', img_eq)

# # Adaptive Equalization
# img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
# utils.showImage('img_adapteq', img_adapteq)

def main():
    filenames = utils.getFilenames()
    images = utils.loadImages(filenames)
    kernel1 = np.ones((3,3), np.uint8)
    kernel9 = np.ones((9,9), np.uint8)

    for index in range(0, len(images)):
        
        image = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        image = diminuirImagem(image)
        # utils.showImage('original', img)

        # Contrast stretching
        p2, p98 = np.percentile(image, (2, 20))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))


        # img = cv2.medianBlur(img_rescale, 25)
        # img = cv2.bilateralFilter(img, 5, 150, 150)
        _,image = cv2.threshold(img_rescale,130,255,cv2.THRESH_BINARY_INV)
        image = cv2.dilate(image, kernel1, iterations=1)
        image=cv2.erode(image,kernel1,iterations=1)
        # img = dog(img)
        image = remove(image,90)
        # img = cv2.medianBlur(img, 5)

        utils.showImage(filenames[index], image)

main()