# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:44:46 2019

@author: prasa
"""
import numpy as np
import cv2
from numpy import linalg as LA
from matplotlib import pyplot as plt

'''
Author: Prasanna Parthasarathy
Color gradients and corresponding histograms:
Gradients (RGB) are calculated by applying Sobel masks and calculating corresponding magnitude and direction at each pixel.
Histograms are based on Gradient orientations.
'''

def normalize(arr):
    min = np.min(arr)
    max = np.max(arr)
    diff = max - min
    for i in range(0,len(arr)):
        arr[i] = (arr[i] - min) * (255 / diff)
    return arr

histograms = []
for i in range(1,100):
    file_index = str(4000 + i)
    file_path_prefix = "../hw2/images/ST2MainHall4/"
    img_name = "ST2MainHall"+file_index+".jpg"
    img_file = file_path_prefix + img_name

    img = cv2.imread(img_file)

    img = cv2.GaussianBlur(img,(15,15),0)

    sobelRx = cv2.Sobel(img[:,:,2],cv2.CV_64F,1,0)
    sobelGx = cv2.Sobel(img[:,:,1],cv2.CV_64F,1,0)
    sobelBx = cv2.Sobel(img[:,:,0],cv2.CV_64F,1,0)
    
    sobelRy= cv2.Sobel(img[:,:,2],cv2.CV_64F,0,1)
    sobelGy = cv2.Sobel(img[:,:,1],cv2.CV_64F,0,1)
    sobelBy = cv2.Sobel(img[:,:,0],cv2.CV_64F,0,1)

    edges = np.abs(sobelRx) + np.abs(sobelGx) + np.abs(sobelBx) + np.abs(sobelRy) + np.abs(sobelGy) + np.abs(sobelBy)
    edges_8u = np.uint8(edges)
    #direction = np.array([sobelRx+sobelGx+sobelBx, sobelRy+sobelGy+sobelBy])
    
    direction = np.arctan2(sobelRx+sobelGx+sobelBx, sobelRy+sobelGy+sobelBy) * 180 / np.pi
    bin_values = np.divide(direction, 10)
    bin_values = np.rint(bin_values)
    val = np.histogram(bin_values.ravel(),36,[0,36])
    histograms.append(val[0])
    plt.plot(val[0])
    plt.show()

plt.clf()

print(len(histograms))


#Histograms intersection
print(histograms[0].shape)
#histogram_intersection = {}
histogram_intersection_scores = np.zeros((99,99),dtype=float)
for x in range(0,99):
    for y in range(x,99):
        num = 0
        den = 0
        for i in range(0,36):
            #print(x,y,i)
            num = num + np.minimum(histograms[x][i], histograms[y][i])
            den = den + np.maximum(histograms[x][i], histograms[y][i])
        intersection = num / den
        #histogram_intersection[(x,y)] = intersection
        histogram_intersection_scores[x][y] = intersection
        histogram_intersection_scores[y][x] = intersection
    

histogram_intersection_scores = normalize(histogram_intersection_scores)
print(len(histogram_intersection_scores))

plt.imshow(histogram_intersection_scores, cmap='hot', interpolation='nearest')
plt.show()
plt.clf()

#Chi-Squared
#chi_squared = {}
chi_squared_scores = np.zeros((99,99),dtype=float)
for x in range(0,99):
    for y in range(x,99):
        chi_squared = 0
        for i in range(0,36):
            if (histograms[x][i] + histograms[y][i]) > 0:
                num = np.square((histograms[x][i] - histograms[y][i]))
                den = histograms[x][i] + histograms[y][i]
                chi_squared = chi_squared + (num / den)
        chi_squared_scores[x][y] = chi_squared
        chi_squared_scores[y][x] = chi_squared

print(len(chi_squared_scores))
chi_squared_scores = normalize(chi_squared_scores)
plt.imshow(chi_squared_scores, cmap='hot', interpolation='nearest')
plt.show()