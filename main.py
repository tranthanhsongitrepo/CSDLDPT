import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

from preprocess import extract_features


def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


def euclidean_distance(ft1, ft2):
    return np.linalg.norm(ft1 - ft2)


def manhattan_distance(p_vec, q_vec):
    """
    This method implements the manhattan distance metric
    :param p_vec: vector one
    :param q_vec: vector two
    :return: the manhattan distance between vector one and two
    """
    return np.sum(np.fabs(p_vec - q_vec))


# reading the image
test_images = 'mẫu 2/2.jpg'
shape = (256, 256)
fig = plt.figure(figsize=(10, 7))

img = cv2.imread('./data/' + test_images)
ft1 = extract_features(img, shape)
rows = 6
columns = 2
# cv2.imshow("", img)
# cv2.waitKey(0)
folders = os.listdir('data')

dsts = []
images = []
for root, dirs, files in os.walk('data'):
    if root == 'data':
        continue
    for name in files:
        img_path = os.path.join(root, name)
        img = cv2.imread(os.path.join(img_path))
        ft2 = extract_features(img, shape, img_path)
        cur_dst = cosine_similarity(ft1, ft2)
        dsts.append(cur_dst)
        images.append(img_path)

images = np.array(images)
dsts = np.array(dsts)

top_ten = images[dsts.argsort()][:11]


for i, image in enumerate(top_ten):
    fig.add_subplot(rows, columns, i + 1)
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[1], shape[0]))
    plt.imshow(img)
    plt.axis('off')
    plt.title(image)
print(top_ten)
plt.show()

"""
['48.jpg' 'anc (1).jpg' '741fc57220b7dcbe6f84c764f99fb8c2.jpg' '45.jpg'
 'n.jpg' '09868fba68d38027db9d12f57242fc4c.jpg' '30.jpg' '19.jpg'
 '02_6_flat.jpg' '49.jpg']
"""
