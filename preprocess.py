# importing required libraries
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _hog(image, shape):
    winSize = shape
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(image, winStride, padding, locations)

    return hist


def _extract_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_cpy = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    image_cpy = cv2.threshold(image_cpy, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    image_cpy = cv2.bitwise_not(image_cpy)
    # Cut just the object out
    sums = image_cpy.sum(axis=0)
    t = np.where(sums != 0)
    x1, x2 = t[0][0], t[0][-1]
    sums = image_cpy.sum(axis=1)
    t = np.where(sums != 0)
    y1, y2 = t[0][0], t[0][-1]

    return image[y1:y2, x1:x2]


def _get_color_mean(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, cnts, -1, 255, -1)

    obj_color_mean = np.array(cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2LAB), mask=mask)[:3])
    obj_color_mean = obj_color_mean.reshape((-1, 1))

    return obj_color_mean


def _pad_resize(image, shape):
    image = cv2.resize(image, (shape[1], shape[0]))
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image


def extract_features(image, shape, name=""):
    assert image.shape[-1] == 3, "Expected 3 channels, got %d" % image.shape[-1]
    image = _extract_object(image)

    # Standardize mean values
    obj_color_mean = _get_color_mean(image)
    obj_color_mean[0] /= 100
    obj_color_mean[1] = (obj_color_mean[1] + 86) / (98 + 86)
    obj_color_mean[2] = (obj_color_mean[2] + 107) / (107 + 94)

    # Pad and resize to get the desired size
    image = _pad_resize(image, shape)
    feature = np.vstack((obj_color_mean, _hog(image, shape)))

    # cv2.imshow("Canny", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return feature

# def extract_features(X, shape):
#     all_features = []
#     for x in X:
#         feature = _extract_features(x, shape)
#         all_features.append(feature)
#     all_features = np.array(all_features)
#     scaler = MinMaxScaler()
#     scaler.fit(all_features)
#     res = scaler.transform(all_features)
#     return res
