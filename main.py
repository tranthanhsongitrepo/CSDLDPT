import cv2
import numpy as np
import os
from preprocess import extract_features

# reading the image
test_images = '08_6_flat.jpg'
shape = (224, 224)

img = cv2.imread('/home/son/PycharmProjects/CSDLDPT/data/' + test_images)
# Find out if the image is bright and adjust clip limit accordingly
# The darker the shirt is the smaller clipLimit should be
ft1 = extract_features(img, shape)
# _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


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
"""['abcd (5).jpg' 'abcd (4).jpg' '4_6_flat.jpg' '25.jpg' 'j.jpg'
 '046_flat (2).jpg' 'l.jpg' 'nh (2).jpg' '27.jpg' '23.jpg']"""
