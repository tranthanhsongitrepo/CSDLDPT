import os
import shutil

imgs1 = os.listdir('/media/son/Games/Images/classroom')
imgs2 = os.listdir('/home/son/Downloads/Annotations/classroom')

names1 = []
for img1 in imgs1:
    names1.append(img1.split('.')[0])

names2 = []
for img2 in imgs2:
    names2.append(img2.split('.')[0])

for img in names1:
    if not img in names2:
        shutil.copyfile('/media/son/Games/Images/classroom/' + img + '.jpg', '/media/son/Games/Images/' + img + '.jpg')
