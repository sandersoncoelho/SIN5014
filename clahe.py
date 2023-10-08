import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize


def showImage(title, image):
  cv2.imshow(title, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

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

# Reading the image from the present directory
image = cv2.imread("../images/m141e diploide.jpg")
# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))
 
# The initial processing of the image
# image = cv2.medianBlur(image, 3)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=5)
clahe_img = clahe.apply(grayImage) + 30
 
_, thresh_img = cv2.threshold(clahe_img, 100, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

blurImage = cv2.GaussianBlur(thresh_img, (19, 19), 5)

# thinImage = cv2.ximgproc.thinning(image, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN)
skeleton = skeletonize(blurImage)
 
# Showing the two images
# cv2.imshow("ordinary threshold", ordinary_img)
showImage("CLAHE image", clahe_img)
    
showImage("thresh image", thresh_img)

showImage("blurImage", blurImage)




# display results
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(skeleton, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()