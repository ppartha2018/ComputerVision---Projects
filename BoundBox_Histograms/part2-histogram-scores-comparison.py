import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

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
    img_file = "images/ST2MainHall4/ST2MainHall"+file_index+".jpg"
    img = cv2.imread(img_file)
    img = np.array(img, dtype=np.int16)
    hist = ((img[:,:,2] >> 5) << 6) + ((img[:,:,1] >> 5) << 3) + (img[:,:,0] >> 5)
    #val = plt.hist(hist.ravel(),512,[0,512])
    val = np.histogram(hist.ravel(),512,[0,512])
    #print(val.shape)
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
        for i in range(0,512):
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
        for i in range(0,512):
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