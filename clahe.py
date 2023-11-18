import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

import basic_filters as bf
from utils import showImage


def differenceOfGaussians(image):
  """Metodo de binarizacao chamado difference of gaussians"""
  g1 = cv2.GaussianBlur(image, (5, 5), 0)
  g2 = cv2.GaussianBlur(image, (15, 15), 0)
  result = g1 - g2
  # showImage('result', result)
  # result = np.where(result > 240, 0, result)
  # showImage('test', test)
  _, image = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
  return image

def main():
  kernel9 = np.ones((9,9), np.uint8)

  # Reading the image from the present directory
  # image = cv2.imread("./dataset/m27e diploide.jpg")
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # image = bf.diminuirImagem(image)
  # showImage('gray image', image)

  #landmark1
  # image = cv2.imread("./landmark1.png")
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # showImage('gray image', image)
  # _, image = cv2.threshold(image,40, 255, cv2.THRESH_BINARY_INV)
  # showImage('gray image', image)
  # image = cv2.medianBlur(image, 5)
  # showImage('gray image', image)

  #landmark10
  image = cv2.imread("./AT-wing-images/AT-0001-031-003678-L.dw.png")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.medianBlur(image, 1)
  showImage('gray image', image)
  _, image = cv2.threshold(image,130, 255, cv2.THRESH_BINARY_INV)
  image = bf.remove(image,90)
  image = cv2.medianBlur(image, 5)
  showImage('gray image', image)
  

  
  # The declaration of CLAHE
  # clipLimit -> Threshold for contrast limiting
  # clahe = cv2.createCLAHE(clipLimit=5)
  # image = clahe.apply(image) + 30
  # _, image = cv2.threshold(image, 155, 255, cv2.THRESH_BINARY_INV)
  # showImage("CLAHE image", image)


  # kernel9 = np.ones((9,9), np.uint8)
  # image = bf.diminuirImagem(image)
  
  # filteredImage = cv2.bilateralFilter(filteredImage, 5, 150, 150)
  # filteredImage = bf.dog(filteredImage)
  # filteredImage = bf.remove(filteredImage,90)
  # filteredImage = cv2.dilate(filteredImage, kernel9, iterations=1)
  # filteredImage=cv2.erode(filteredImage,kernel9,iterations=1)
  # filteredImage=bf.removeWings(filteredImage,kernel9)

  # xx,ww,hh,yy = bf.cutImage(filteredImage) 
  # filteredImage = filteredImage[xx:yy, ww:hh]


  # image = cv2.medianBlur(clahe_img, 9)
  # image = cv2.bilateralFilter(image, 5, 150, 150)
  # image = bf.dog(clahe_img)
  # image = bf.remove(image,90)
  # image = cv2.dilate(image, kernel9, iterations=1)
  # image=cv2.erode(image,kernel9,iterations=1)
  # image=bf.removeWings(image,kernel9)
  # xx,ww,hh,yy = bf.cutImage(image) 
  # image = image[xx:yy, ww:hh]
  image = cv2.ximgproc.thinning(image,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
  showImage("image final", image)
  

  # image = cv2.GaussianBlur(image, (1, 1), 5)
  # image = bf.remove(image,90)

  # image = bf.dog(image)

  # thinImage = cv2.ximgproc.thinning(image, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN)
  
  # Showing the two images
  # cv2.imshow("ordinary threshold", ordinary_img)
  
  
      
  # showImage("thresh image", thresh_img)




  # display results
  # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4),
  #                         sharex=True, sharey=True)

  # ax = axes.ravel()

  # ax[0].imshow(skeleton, cmap=plt.cm.gray)
  # ax[0].axis('off')
  # ax[0].set_title('skeleton', fontsize=20)

  # fig.tight_layout()
  # plt.show()

main()