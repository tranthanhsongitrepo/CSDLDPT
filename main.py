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
    return np.sum(np.fabs(p_vec - q_vec))


data_folder = "data"
test_images_path = os.path.join(data_folder, 'mẫu 5', '1.jpg')
shape = (256, 256)
fig = plt.figure(figsize=(10, 7))

test_img = cv2.imread(test_images_path)
ft1 = extract_features(test_img, shape)
rows = 6
columns = 2

dsts = []
images = []
for root, dirs, files in os.walk(data_folder):
    if root == data_folder:
        continue
    for name in files:
        img_path = os.path.join(root, name)
        img = cv2.imread(img_path)
        ft2 = extract_features(img, shape, img_path)
        cur_dst = cosine_similarity(ft1, ft2)
        dsts.append(cur_dst)
        images.append(img_path)

images = np.array(images)
dsts = np.array(dsts)

top_ten = images[dsts.argsort()][1:11]

fig.add_subplot(rows, columns, 1)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (shape[1], shape[0]))
plt.imshow(test_img)
plt.axis('off')
plt.title(test_images_path)

for i, image in enumerate(top_ten):
    fig.add_subplot(rows, columns, i + 3)
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[1], shape[0]))
    plt.imshow(img)
    plt.axis('off')
    plt.title(image)

print(top_ten)
plt.show()
