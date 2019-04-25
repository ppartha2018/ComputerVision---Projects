import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
Quiver plots for displaying edge orientations in an image:
Calculate Edge orientations with Sobel masks method, choose orientations at uniformly spaced intervals
and feed it into matplotlib's quiver method to get edge orientations corresponding to an image.
'''

def displayImage(img):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_file = "../hw2/images/ST2MainHall4/ST2MainHall4086.jpg"
#img_file = "../hw2/images/rgbgy.png"
#img_file = "../hw2/images/background.jpg"

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
print("Input image shape: ", img.shape)
print("Input image type: ", type(img[0]))

displayImage(img)
img = cv2.GaussianBlur(img,(5,5),0)

displayImage(img)

sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0)
#abs_sobelx64f = np.absolute(sobelx64f)
#sobelx_8u = np.uint8(abs_sobelx64f)
sobelx_8u = np.uint8(sobelx64f)

sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1)
#abs_sobely64f = np.absolute(sobely64f)
#sobely_8u = np.uint8(abs_sobely64f)
sobely_8u = np.uint8(sobely64f)

edges = np.abs(sobelx64f) + np.abs(sobely64f)

magnitude, direction = cv2.cartToPolar(sobelx64f,sobely64f, angleInDegrees=True)

bin_values = np.divide(direction, 10)
bin_values = np.rint(bin_values)
val = np.histogram(bin_values.ravel(),36)
plt.plot(val[0])
plt.show()

magnitude_8u = np.uint8(magnitude)
#edges = np.sqrt(np.square(sobelx64f) + np.square(sobely64f))
edges_8u = np.uint8(edges)

print(edges_8u.shape)
print(magnitude.shape)
#displayImage(edges_8u)
displayImage(magnitude_8u)
#cv2.imwrite("EdgeGrayGradients_ST2MainHall4001.png", magnitude_8u)

X,Y = np.meshgrid(np.arange(1600), np.arange(1200))
fig1, ax1 = plt.subplots()
ax1.set_title("Edge Orientation")
Q = ax1.quiver(X[::40,::40], Y[::40,::40], sobelx_8u[::40,::40], sobely_8u[::40,::40],units="width")
plt.figure()
plt.show()