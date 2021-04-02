from skimage import io
from skimage.feature import haar_like_feature
from skimage.transform import integral_image

img = io.imread('/home/son/PycharmProjects/CSDLDPT/preprocessed/12386489MT_13_n_r.jpg')

ii = integral_image(img)

features = haar_like_feature(ii, 0, 0, ii.shape[1], ii.shape[0], 'type-2-x')
print(features)