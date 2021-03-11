import cv2
import os
img = cv2.imread('/datasets/Yemek\\Tavuk_Corbasi\\img4.png')

print('Original Dimensions : ', img.shape)

width = 100
height = 50  # keep original height
dim = (width, height)
print(dim)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', resized.shape)
cv2.imshow("Resized image", img)

cv2.waitKey(0)