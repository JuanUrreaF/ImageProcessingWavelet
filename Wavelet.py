# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:09:06 2020

@author: user
"""
import cv2
import pywt
import matplotlib.pyplot as plt
import numpy as np

I_Pre = cv2.imread("PREPROC_HMUA017_AFTER.tif",cv2.cv2.IMREAD_GRAYSCALE)
#plt.gray()

"""
coeffs2 = pywt.dwt2(I_Pre,"db1",mode = "periodization")
cA, (cH,cV,cD) = coeffs2

Im = pywt.idwt2(coeffs2,"db1", mode="periodization")
Im = np.uint8(Im)

plt.figure(figsize=(30,30))

plt.subplot(2,2,1)
plt.imshow(cA, cmap=plt.cm.gray)
plt.subplot(2,2,2)
plt.imshow(cH, cmap=plt.cm.gray)
plt.subplot(2,2,3)
plt.imshow(cV, cmap=plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(cD, cmap=plt.cm.gray)
"""

"""
cH = np.resize(cH, (921,1102))
cD = np.resize(cD, (921,1102))

Sum = I_Pre + cD

plt.figure(2)
plt.imshow(I_Pre,cmap=plt.cm.gray)

plt.figure(3)
plt.imshow(Sum,cmap=plt.cm.gray)

#print(pywt.families(short=False))
"""

C = pywt.wavedec2(I_Pre, "db1", mode = "periodization", level = 2)
Imgr = pywt.waverec2(C, "db1", mode = "periodization")
Imgr = np.uint8(Imgr)

cA2 = C[0]
(cH1, cV1, cD1) = C[-1]
(cH2, cV2, cD2) = C[-2]

plt.figure(figsize=(30,30))

plt.subplot(2,2,1)
plt.imshow(cA2, cmap=plt.cm.gray)
plt.subplot(2,2,2)
plt.imshow(cH2, cmap=plt.cm.gray)
plt.subplot(2,2,3)
plt.imshow(cV2, cmap=plt.cm.gray)
plt.subplot(2,2,4)
plt.imshow(cD2, cmap=plt.cm.gray)

arr, coeff_slices = pywt.coeffs_to_array(C)
plt.figure(figsize=(30,30))
plt.imshow(arr, cmap=plt.cm.gray)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(Imgr, cmap=plt.cm.gray)
plt.title("Wavelet")
plt.subplot(1,2,2)
plt.imshow(I_Pre)
plt.title("Original")

plt.figure()
plt.hist(cA2.ravel(),256,[0,256])
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imread 
from mpl_toolkits.mplot3d import Axes3D 
import scipy.ndimage as ndimage 
 
mat = I_Pre
mat = mat[:,:]
rows, cols = mat.shape 
xv, yv = np.meshgrid(range(cols), range(rows)[::-1]) 
 
blurred = ndimage.gaussian_filter(mat, sigma=(5, 5), order=0) 
fig = plt.figure(figsize=(6,6)) 
 
ax = fig.add_subplot(121) 
ax.imshow(mat, cmap='gray') 
 
ax = fig.add_subplot(122, projection='3d') 
ax.elev= 75 
ax.plot_surface(xv, yv, mat) 
 
# ax = fig.add_subplot(223) 
# ax.imshow(blurred, cmap='gray') 
 
# ax = fig.add_subplot(224, projection='3d') 
# ax.elev= 75 
# ax.plot_surface(xv, yv, blurred) 
plt.show()

_,Binarizada = cv2.threshold(Imgr,126,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(6)
plt.gray()
plt.imshow(Binarizada)

plt.figure(7)
plt.imshow(I_Pre)