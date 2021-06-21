import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

import pandas as pd
from preprocess import extract_features, to_csv


def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


def euclidean_distance(ft1, ft2):
    return np.linalg.norm(ft1 - ft2)


def manhattan_distance(p_vec, q_vec):
    return np.sum(np.fabs(p_vec - q_vec))


# to_csv('data.csv')

train_folder = 'data/train'
test_images_path = 'sơ mi dài tay cổ trụ 1 túi_95160aa2faabbfaff4290f7e33af3ab9.jpg'
df = pd.read_csv('data.csv')
shape = (256, 256)
fig = plt.figure(figsize=(10, 7))
rows = 6
columns = 2
test_features = df.loc[df['name'] == test_images_path]
fig.add_subplot(rows, columns, 1)


dsts = []
images = []

for idx, row in df.iterrows():
    dsts.append(cosine_similarity(row[2:].to_numpy(), test_features.to_numpy().T[2:].reshape(-1)))
    images.append(row[1])

dsts = np.array(dsts)
images = np.array(images)

top_ten = images[dsts.argsort()][1:11]

test_img = cv2.imread(os.path.join(train_folder, test_images_path))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (shape[1], shape[0]))
plt.imshow(test_img)
plt.axis('off')
plt.title(test_images_path)

for i, image in enumerate(top_ten):
    fig.add_subplot(rows, columns, i + 3)
    img = cv2.imread(os.path.join(train_folder, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape[1], shape[0]))
    plt.imshow(img)
    plt.axis('off')
    plt.title(image)

print(top_ten)

plt.show()
