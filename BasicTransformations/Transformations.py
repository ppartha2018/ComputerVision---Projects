import cv2
from matplotlib import pyplot as plt
import numpy as np

input_image = "Prasanna.png"

def display_save(display_name, file_name, img):
    cv2.imshow(display_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(file_name, img)

def grayscale():
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    display_save("Gray", "prasanna_gray.png", img)

def blur():
    img = cv2.imread(input_image)
    kernel = np.ones((20,20),np.float32)/400
    blur = cv2.filter2D(img,-1,kernel)
    display_save("Blurred", "prasanna_blurred.png", blur)

def change_color_space():
    img = cv2.imread(input_image)
    op = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    display_save("BGR2HSV", "prasanna_hsv.png", op)

def mix_color_scheme():
    img = cv2.imread(input_image)
    rows,cols,channels = img.shape
    img[:int(rows),:2*int(cols/3),:] = cv2.cvtColor(img[:int(rows),:2*int(cols/3),:],cv2.COLOR_BGR2RBG)
    img[:int(rows),int(cols/3):2*int(cols/3),:] = cv2.cvtColor(img[:int(rows),:int(cols/3):2*int(cols/3),:],cv2.COLOR_LAB2BGR)
    img[:int(rows),2*int(cols/3):,:] = cv2.cvtColor(img[:int(rows),2*int(cols/3):,:],cv2.COLOR_RGB2YCrCb)
    display_save("Mix Color", "prasanna_mix_color.png", img)

def erosion():
    img = cv2.imread(input_image)
    kernel = np.ones((8,8),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    display_save("erosion", "prasanna_eroded.png", erosion)

def morph_gradient():
    img = cv2.imread(input_image)
    kernel = np.ones((10,10),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    display_save("gradient", "prasanna_gradient.png", gradient)

def mask_face():
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[200:780, 300:750] = 255
    masked_img = cv2.bitwise_and(img,img,mask = mask)
    display_save("Masked Image", "masked_img.png", masked_img)

def histogram_equalization():
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    display_save("Histogram Equalization", "prasanna_gray_histequ.png", equ)

def affine_transformation():
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    rows,cols = map(int, img.shape)
    pts1 = np.float32([[70,50],[200,70],[70,200]])
    pts2 = np.float32([[100,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    display_save("Affline Transformation", "prasanna_affline1.png", dst)

def edge_detection():
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img,100,200)
    display_save("Edges", "prasanna_edges.png", edges)

def image_thresholding():
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    display_save("IMG_THRESHOLDING", "prasanna_thresholding.png", thresh1)

'''
grayscale()
blur() #1
change_color_space() #2
switch_color_scheme() #3
erosion() #4
morph_gradient() #5
mask_face() #6
histogram_equalization() #7
affine_transformation() #8
edge_detection() #9
image_thresholding() #10
'''
blur()