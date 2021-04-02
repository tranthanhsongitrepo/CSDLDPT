import cv2
import numpy as np
import os
from preprocess import extract_features

# reading the image
test_images = '08_6_flat.jpg'
shape = (64, 64)

img = cv2.imread('/home/son/PycharmProjects/CSDLDPT/data/' + test_images)
ft1 = extract_features(img, shape)

# cv2.imshow("", img)
# cv2.waitKey(0)
images = np.array(os.listdir('data'))
dst = np.zeros(len(images))
i = 0
for image in images:
    img = cv2.imread(os.path.join('data', image))
    ft2 = extract_features(img, shape, image)
    dst[i] = np.linalg.norm(ft1 - ft2)
    i += 1

print(images[dst.argsort()][1:11])

