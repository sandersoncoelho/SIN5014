import glob
import sys

import config
import cv2
import numpy as np


def showImage(title, image):
  cv2.imshow(title, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def dog(img):
  """Metodo de binarizacao chamado difference of gaussians"""
  g1 =  cv2.GaussianBlur(img,(1,1),0)
  g2=  cv2.GaussianBlur(img,(3,3),0)
  result = g1 - g2
  _,img = cv2.threshold(result,128,255,cv2.THRESH_BINARY)
  return img

def remove(img, Min):
    """Recebe uma imagem e um tamanho minimo, devolve a imagem apenas com os objetos maiores que o tamanho minimo"""
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1] 
    nb_components = nb_components - 1
    min_size = Min
    img = np.zeros((output.shape),np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    return img

def find_nearest_white(img, target):
    """Da uma localizacao e devolve a localizacao do ponto branco mais proximo, sera usado para manter apenas a asa centralizada"""
    img = remove(img, 2000)
    nonzero2 = cv2.findNonZero(img)           
    distances = np.sqrt((nonzero2[:,:,0] - target[1]) ** 2 + (nonzero2[:,:,1] - target[0]) ** 2)
    nearest_index = np.argmin(distances)
    nearest_white= nonzero2[nearest_index]
    return nearest_white

def removewings(img,kernel):
    """Metodo que deixa apenas a asa centralizada"""
    """Dropbox - > ferias -> Tirarruidosenormes2 explica o algoritmo""" 

    big = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    target = int(len(big)/2) , int(len(big[0])/2)
    TARGET = find_nearest_white(big,target)
    nb_components2, output2, stats2, centroids2 = cv2.connectedComponentsWithStats(big, connectivity=8)
    nb_components2 = nb_components2 - 1
    imga = np.zeros((output2.shape),np.uint8)
    booleano = 0
    for i in range(0, nb_components2):
        if booleano == 0 :
            imgtransit = np.zeros((output2.shape),np.uint8)
            imgtransit[output2 == i + 1] = 255
            if imgtransit[TARGET[0][1]][TARGET[0][0]] == 255:
                imga[output2 == i + 1] = 255
                booleano == 1
    img_bwo2 = cv2.bitwise_and(imga, img)
    return img_bwo2

def diminuirimagem(img):
    """Diminui a imagem para determinada quantidade de pixels (se ela for maior que essa quantidade)"""
    # x=((pixels)/(len(img[0])*len(img)))**0.5
    # if x<1 :
    #     print('asdf:', pixels, len(img[0]), len(img))
    #     img = cv2.resize(img,(int(len(img[0])*x),int(len(img)*x)), interpolation=cv2.INTER_AREA)
    # return img
    (height, width) = img.shape[:2]
    # print('size:',height,width)
    if height * width > 480000:
        heightFinal = 600
        widthFinal = (heightFinal * width) / height
        imgfinal = cv2.resize(img, (int(widthFinal), int(heightFinal)), interpolation=cv2.INTER_AREA)
        return imgfinal
    return img
            

def cutimage(img):
    """Devolve o bounding box da asa binarizada xx,ww,hh,yy
    Alem disso, zcolsums e uma variavel que possui a soma dos pixels de cada coluna da matriz da imagem,
    e uma maneira de identificar se a asa esta inteira na imagem ou esta cortada, pois se a primeira coluna for = 0, significa que 
    nao existe pixels da asa nessa parte inicial da matriz.
    """
    zcolsums = np.sum(img, axis=0)
    zlines = np.sum(img, axis=1)
    zcolsums2 = np.nonzero(0-zcolsums)                                                                 
    zlines2 = np.nonzero(0-zlines)    
    xx=zlines2[0][0]                                                               
    yy=zlines2[0][zlines2[0].shape[0]-1]    
    ww=zcolsums2[0][0]
    hh=zcolsums2[0][zcolsums2[0].shape[0]-1]
    if xx > 7 :
        xx = xx-8
    else :
        xx = 0 
        
    if ww > 7 :
        ww = ww-8
    else :
        ww = 0
    
    if hh < img.shape[1] -9:
        hh=hh+8
    else :
        hh=img.shape[1]-1
        
    if yy < img.shape[0] -9:
        yy=yy+8
    else :
        yy=img.shape[0]-1
    return xx,ww,hh,yy

def getTranslationMatrix2d(dx, dy):
    """
    Funcao auxiliar para rotacionar a imagem
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

def rotateImage(image, angle):
    """
    Dado um angulo, rotaciona a imagem
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    return result

def rotacionarfinal(img):
    """recebe uma imagem, encontra o algulo ideal para rotacionar e rotaciona, e devolve o angulo usado"""
    menoresquerda=0
    menordireita=0
    encontrou = 0

    for y in range(int(len(img[0]))):
        if encontrou == 1:
            break
        for x in range(int(len(img))):
            if img[x][int(len(img[0])*0.9)-1-y]==255:
                menordireita=np.array([x,int(len(img[0])*0.9)-1-y])
                encontrou = 1
                break
       
    encontrou=0    
    for y in range(int(len(img[0])*0.1),int(len(img[0]))):
        if encontrou == 1:
            break
        for x in range(int(len(img))):
            if img[x][y]==255:
                menoresquerda=np.array([x,y])
                encontrou = 1     
                break

    b=menordireita
    a=np.array([menordireita[0],menoresquerda[1]])
    c=menoresquerda 
    ba = a - b
    bc = c - b      
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle)*-1)
    
    if menoresquerda[0]<menordireita[0]:
        angle=angle*-1
    if menoresquerda[0]!=menordireita[0]:
        img = rotateImage(img, angle)
    return img  , angle 

def getwhitepixelscoordinates(img):
    """Recebe uma imagem binaria e retorna os coordenadas dos pixels brancos"""
    nonzero2 = cv2.findNonZero(img)
    nonzero2[0][0][0]
    whitepixels = nonzero2[:, 0, :]    
    return whitepixels

def getLandmarks(input_image):
    """Metodo que recebe uma imagem esquelitzada e devolve uma lista contendo os pontos de interesse"""
   
    """Templetes de interesse"""
    kernel = np.array((
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]), dtype="int")

    kernel2 = np.array((
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]), dtype="int")
    
    kernel3 = np.array((
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]), dtype="int")
    
    kernel4 = np.array((
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0]), dtype="int")
    
    kernel5 = np.array((
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 1]), dtype="int")
    
    kernel6 = np.array((
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 0]), dtype="int")
    
    kernel7 = np.array((
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0]), dtype="int")
    
    kernel8 = np.array((
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1]), dtype="int")
    
    kernel9 = np.array((
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1]), dtype="int")
    
    kernel10 = np.array((
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0]), dtype="int")
    
    kernel11 = np.array((
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1]), dtype="int")
    
    kernel12 = np.array((
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1]), dtype="int")
    
    """Unindo todos pontos de interesse"""
    output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
    output_image2 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel2)
    output_image3 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel3)
    output_image4 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel4)
    output_image5 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel5)
    output_image6 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel6)
    output_image7 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel7)
    output_image8 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel8)
    output_image9 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel9)
    output_image10 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel10)
    output_image11 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel11)
    output_image12 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel12)
    output_image = cv2.bitwise_or(output_image,output_image2)
    output_image2 = cv2.bitwise_or(output_image3,output_image4)
    output_image3 = cv2.bitwise_or(output_image5,output_image6)
    output_image4 = cv2.bitwise_or(output_image7,output_image8)
    output_image5 = cv2.bitwise_or(output_image9,output_image10)
    output_image6 = cv2.bitwise_or(output_image11,output_image12)
    output_image8 = cv2.bitwise_or(output_image,output_image2)
    output_image9 = cv2.bitwise_or(output_image3,output_image4)
    output_image10 = cv2.bitwise_or(output_image5,output_image6)
    output_image11 = cv2.bitwise_or(output_image8,output_image9)
    output_image12 = cv2.bitwise_or(output_image10,output_image11)
    i,j = np.where(output_image12 == 255)
    intersections=[]
    
    """Caso a asa esteja inteira na imagem, excluir os pontos de interesse perto das bordas laterais"""
    zcolsums = np.sum(input_image, axis=0)
    for x in range(i.shape[0]):

        if zcolsums[0]==0 and zcolsums[zcolsums.shape[0]-1]==0:
            lateral1=int(output_image12.shape[1]/5)
            lateral2=int(output_image12.shape[1]/8)

        else :
            lateral1=int(output_image12.shape[1]/25)
            lateral2=int(output_image12.shape[1]/25)

            
        if j[x] > lateral1 and i[x]< output_image12.shape[0]-2  and i[x]>1 and j[x]< output_image12.shape[1]-lateral2:
            intersections.append((j[x],i[x]))

    """Se tiver dois pontos de interesse muito pertos, deixar apenas um"""


    return intersections

def finalintersec(intersec,intersec2):
    """Pega os pontos de interesse da asafinal1 e da asafinal2, e realiza a interseccao"""
    intersecc=[]    
    for point1 in intersec2:
        point=calculate_distance(point1,intersec)
        if point !=(0,0):
            if point not in intersecc:
                intersecc.append(point)    

    for point1 in intersecc:
        for point2 in intersecc:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 12**2) and (point1 != point2):
                intersecc.remove(point2)
    if len(intersecc)==0 and len(intersec)!=0 :
        intersecc= intersec.copy()
  
    elif len(intersecc)==0 and len(intersec)==0 and len(intersec2)!=0 :
        intersecc= intersec2.copy()
   
    elif len(intersecc)==0 and len(intersec)==0 and len(intersec2)==0:   
        intersecc.append((0,0))
    return intersecc

def calculate_distance(ponto1,list):
    "Calcula distancia de dois pontos"
    menor=11
    ponto=(0,0)
    for ponto2 in list :
        dist = ( (ponto2[0] - ponto1[0])**2 + (ponto2[1] - ponto1[1])**2 )**0.5
        if(dist<menor):
            ponto=ponto2
            menor=dist
    return ponto

def mylistcolor(image, dataset,coordinates,weight):
    """Metodo que acrescenta os valores das cores nos pixels ja localizados"""
    
    """Input:
        Image : imagem colorida
       dataset: localizacao dos pixels dessa asa
       coordinates: localizacao dos pixels de interesse dessa asa (eventualmente todos pixels da asa sao pixels de interesse, logo 
       dataset=coordinates)
       
       Output:
         devolve as coordinates, acrescentando o valor das cores de cada pixel  

    """

    my_list = []

    colora=[]
    colorb=[]
    for x in range(len(dataset)):
        colora.append(image[dataset[x][1]][dataset[x][0]][0])
        colorb.append(image[dataset[x][1]][dataset[x][0]][1])

    mincolora=min(colora)
    mincolorb=min(colorb)


    colora=colora-mincolora
    colorb=colorb-mincolorb
    
    maxcolora=max(colora)
    maxcolorb=max(colorb)
    if len(dataset) != len(coordinates):
        colora=[]
        colorb=[]
        for x in range(len(coordinates)):
            colora.append(image[coordinates[x][1]][coordinates[x][0]][0])
            colorb.append(image[coordinates[x][1]][coordinates[x][0]][1])
        colora=colora-mincolora
        colorb=colorb-mincolorb

    colora=colora/maxcolora*weight
    colorb=colorb/maxcolorb*weight
    colora = np.array(colora)
    colorb = np.array(colorb)
    colora= colora.reshape((len(colora),1))
    colorb= colorb.reshape((len(colorb),1))
    color = np.concatenate((colora,colorb), axis=1)
    my_list = np.concatenate((coordinates,color), axis=1)
    return my_list

def applyFilters(image):
    kernel9 = np.ones((9,9), np.uint8)

    _image = diminuirimagem(image)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    _image = cv2.medianBlur(_image, 5)
    _image = cv2.bilateralFilter(_image, 5, 150, 150)
    _image = dog(_image)
    _image = remove(_image,90)
    _image = cv2.dilate(_image, kernel9, iterations=1)
    _image=cv2.erode(_image,kernel9,iterations=1)
    _image=removewings(_image,kernel9)

    xx,ww,hh,yy = cutimage(_image) 
    _image = _image[xx:yy, ww:hh]

    _image = cv2.ximgproc.thinning(_image,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    return _image

def locateLandmarks(image):
    landmarks = getLandmarks(image)

    for p in landmarks:
        x = p[0]
        y = p[1]
        image = cv2.circle(image, (x, y), 5, 255, -1)
    
    return image, landmarks

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

def saveLandmarks(landmarks, filename):
    path = filename.replace(config.DATASET_PATH, config.OUT_PATH + '/landmarks')
    path = path.replace('.' + config.DATASET_IN_EXTENSION, '')
    np.save(path, landmarks)

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
