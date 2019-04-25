import sys
import cv2
import math
import numpy as np

'''
Author: Prasanna Parthasarathy
This script constructs all layers of gaussian pyramid from the input image and packs into "samllest possible" rectangle,
irrespective of the input dimensions. (i.e uneven, row-wise or column-wise).

'''

input = cv2.imread("Prasanna.png")

rows, cols, channels = map(int, input.shape)
print(rows, cols, channels)

#store the original size - will need for packing later
input_h, input_w = rows, cols

#number of times till the top of pyramid
pyramid_height = int(math.log(max(rows,cols),2))

pyramid_layers = []
pyramid_layers.append(input)
for i in range(0, pyramid_height):
    rows, cols, channels = map(int, input.shape)
    input = cv2.pyrDown(input, dstsize=(cols // 2, rows // 2))
    pyramid_layers.append(input)

#the smallest possible rectangle we need is [ max(input_h, input_w) * (min(input_h, input_w) + min(input_h, input_w)/2) ]
stack_sideways = True
if max(input_h, input_w) == input_h:
    pack_input_h = input_h + int(input_h / 2)
    pack_input_w = input_w
    displacement_h = input_h
    displacement_w = 0
else:
    pack_input_w = input_w + int(input_w / 2) 
    pack_input_h = input_h
    displacement_h = 0
    displacement_w = input_w
    stack_sideways = False #stack to bottom

#h1, w1 = pyramid_layers[0].shape[:2]
#h2, w2 = pyramid_layers[1].shape[:2]
print("Input size: ", input_h, input_w)
print("Packing size: ", pack_input_h, pack_input_w)
result = np.zeros((pack_input_h, pack_input_w, 3), np.uint8)

#pack input image
result[:input_h, :input_w,:3] = pyramid_layers[0]

#rest all the other layers in the remaining space
for i in range(1,len(pyramid_layers)):
    hi, wi = pyramid_layers[i].shape[:2]
    print(pack_input_h, pack_input_w, displacement_h, displacement_w, hi, wi)
    if stack_sideways:
        result[displacement_h:displacement_h+hi, displacement_w:displacement_w+wi,:3] = pyramid_layers[i]
        displacement_w = displacement_w+wi
    else:
        result[displacement_h:displacement_h+hi, displacement_w:displacement_w+wi,:3] = pyramid_layers[i]
        displacement_h = displacement_h+hi

cv2.imshow("Image Pyramid", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("prasanna_gaussian_pyramid.png", result)