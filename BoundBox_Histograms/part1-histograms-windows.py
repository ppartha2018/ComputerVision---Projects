import cv2
import numpy as np
from tkinter import filedialog
from tkinter import *

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("png,tiff,bmp,jpeg files","*.tif*"),("all files","*.*")))
root.destroy()

img_name = root.filename
img = cv2.imread(img_name)
imgY,imgX,imgC = img.shape
orig = np.array(img)

'''
Author: Prasanna Parthasarathy
Load an image and compute the bounding box aroud the mouse pointer.
Computer the window statistics and display it in separate window.
calc_histogram_color and plot_histogram - deals with reading a seqeunce of images, computing histograms and performing comparisons.
'''

#Naive - O(n^2) algorithm. Better implementation is done with np arrays for Part2. 
def calc_histogram_color(img):
    x,y,c = img.shape
    histograms = []
    for band in range(0,c):
        histogram = np.zeros(256);
        for i in range(0,x):
            for j in range(0,y):
                val = img[i][j][band]
                histogram[val] = histogram[val] + 1
        histograms.append(histogram)
    return histograms

def plot_histogram(hist):
    x,y,c = img.shape
    print("Image dimensions: ", x,y,c)
    print("No of pixels: ", (x*y))
    color = {'blue','green','red'}
    for i,col in enumerate(color):
        plt.plot(hist[i], color=col, label=col)
        plt.legend()
        plt.show()
        plt.clf()
        plt.show()

hist = calc_histogram_color(img)
plot_histogram(hist)
  
'''
Display the image, on muouse hover event, display the outer border based on the x,y as center of the window.
Display the statistics of the window in 2 separate windows.
'''
def image_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img = np.array(orig)
        
        #print("Mouse at: ", x,y)
        #11*11 outer border => 13*13 border
        borderX = 13
        borderY = 13
        #display gray box in outer border
        xMin = x - (borderX >> 1) if x - (borderX >> 1) >= 0 else 0
        xMax = x + (borderX >> 1) if x + (borderX >> 1) < imgX else imgX-1
        yMin = y - (borderY >> 1) if y - (borderY >> 1) >= 0 else 0
        yMax = y + (borderY >> 1) if y + (borderY >> 1) < imgY else imgY-1
        #print("Top Corner at: ", xMin,yMin)
        for i in range(xMin,xMax+1):
            img[yMin][i] = 255
            img[yMax][i] = 255
        for j in range(yMin,yMax+1):
            img[j][xMin] = 255
            img[j][xMax] = 255
        topBoxX = xMin - 120
        topBoxY = yMin - 120
        topBox = np.zeros((30,600,3), np.uint8)
        bottomBox = np.zeros((30,400,3), np.uint8)
        
        theBox = img[yMin+1:yMax,xMin+1:xMax]
        
        #can be improved for efficiency
        topBoxContent = "Mouse at Pixel(X,Y): (" + str(x) + "," + str(y) + ") Color Values (B G R): " + str(img[y][x])
        r,g,b = int(img[y][x][2]), int(img[y][x][1]), int(img[y][x][0]) 
        intensity  = (r + g + b) / 3
        #print("Intensity: " + intensity)
        bottomBoxContent = "Intensity: " + str(round(intensity,2)) + " Mean: " + str(round(np.mean(theBox),2)) + " SD: " + str(round(np.std(theBox),2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        textStart = (10,20)
        fontScale = 0.5
        fontColor = (255,255,255)
        lineType = 1

        cv2.putText(topBox,topBoxContent, textStart, font, fontScale,fontColor, lineType)
        cv2.putText(bottomBox,bottomBoxContent, textStart, font, fontScale,fontColor, lineType)
        
        cv2.imshow("Input Image", img)
        cv2.imshow("Top Box", topBox)
        cv2.imshow("Bottom Box", bottomBox)
        

def show_image(img):
    
    cv2.imshow("Input Image", img)
    cv2.setMouseCallback("Input Image", image_callback)        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
show_image(img)