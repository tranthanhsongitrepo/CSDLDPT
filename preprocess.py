# importing required libraries
import cv2
import numpy as np
import os
import pandas as pd


def _hog(image, shape):
    block_size = 16
    cell_size = 8
    assert (image.shape[0] % cell_size == 0 and image.shape[1] % cell_size == 0), "Size not supported"
    nbins = 9
    dx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    dy = dx.T
    # tinh -1: độ sâu
    gx = cv2.filter2D(image, -1, dx)
    gy = cv2.filter2D(image, -1, dy)

    #
    gs = np.sqrt(np.square(gx) + np.square(gy))
    phis = np.arctan(gy / (gx + 1e-6))
    phis[gx == 0] = np.pi / 2

    argmax_g = gs.argmax(axis=-1)

    # lấy ra g, phi mà tại đó g max
    g = np.take_along_axis(gs, argmax_g[..., None], axis=1)[..., 0]
    phi = np.take_along_axis(phis, argmax_g[..., None], axis=1)[..., 0]
    histogram = np.zeros((g.shape[0] // cell_size, g.shape[1] // cell_size, nbins))
    for i in range(0, g.shape[0] - cell_size + 1, cell_size):
        for j in range(0, g.shape[1] - cell_size + 1, cell_size):
            g_in_square = g[i:i + cell_size, j:j + cell_size]
            phi_in_square = phi[i:i + cell_size, j:j + cell_size]

            bins = np.zeros(9)

            for u in range(0, g_in_square.shape[0]):
                for v in range(0, g_in_square.shape[1]):
                    g_pixel = g_in_square[u, v]
                    phi_pixel = phi_in_square[u, v] * 180 / np.pi
                    bin_index = int(phi_pixel // 20)
                    a = bin_index * 20
                    b = (bin_index + 1) * 20

                    value_1 = (phi_pixel - a) / 20 * g_pixel
                    value_2 = (b - phi_pixel) / 20 * g_pixel

                    bins[bin_index] += value_2
                    bins[(bin_index + 1) % 9] += value_1

            histogram[int(i / cell_size), int(j / cell_size), :] = bins

    t = block_size // cell_size
    hist = []
    for i in range(0, histogram.shape[0] - t + 1):
        for j in range(0, histogram.shape[1] - t + 1):
            block = histogram[i:i + t, j:j + t, :]
            block = block.flatten()
            block /= np.linalg.norm(block) + 1e-6
            hist.append(block)

    hist = np.array(hist)

    return hist.flatten()


# def _hog(image, shape):

    # return hist

# ['sơ mi dài tay trơn 1 màu cổ đức_7594dfd43674e654fde2305dd37ceab8.jpg'
#  'sơ mi dài tay không cổ_3.jpg'
#  'sơ mi dài tay cổ trụ 1 túi_f676705fee6b6c34aa7e89d1b2a772cc.jpg'
#  'sơ mi dài tay cổ trụ 1 túi_95160aa2faabbfaff4290f7e33af3ab9.jpg'
#  'sơ mi dài tay cổ trụ 1 túi_3daf093650dab029b077be41ab9ac090.jpg'
#  'sơ mi dài tay kẻ caro 1 túi_479aa6dc55d1e4b3761dc854311f8c25.jpg'
#  'sơ mi dài tay kẻ caro 1 túi_05522587fd0c8ca31eea976732eaa805.jpg'
#  'sơ mi dài tay trơn 1 màu cổ đức_4b1330ab45aafb13c4a8097980fb6cb8.jpg'
#  'sơ mi dài tay cổ trụ 1 túi_ef15008bd0d10745323d6a805dcd6052.jpg'
#  'sơ mi dài tay trơn 1 màu cổ đức_018da9dea2cdbbd64c20d2cf3efa2e1c.jpg']

def _extract_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_cpy = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    image_cpy = cv2.bitwise_not(image_cpy)

    # Cut just the object out
    sums = image_cpy.sum(axis=0)
    t = np.where(sums != 0)
    x1, x2 = t[0][0], t[0][-1]
    sums = image_cpy.sum(axis=1)
    t = np.where(sums != 0)
    y1, y2 = t[0][0], t[0][-1]

    return image[y1:y2 + 1, x1:x2 + 1]


def _get_color_mean(image):
    # Lấy trung bình 3 từng kênh L*, a*, b*
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    obj_color_mean = np.array(lab.mean(axis=(0, 1))[:3])
    obj_color_mean = obj_color_mean.reshape((-1, 1))

    return obj_color_mean


def _get_color_std(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    obj_color_std = np.array(lab.std(axis=(0, 1))[:3])
    obj_color_std = obj_color_std.reshape((-1, 1))

    return obj_color_std


def _pad_resize(image, shape):
    image = cv2.resize(image, (shape[1], shape[0]))
    return image


def extract_features(image, shape, name=""):
    assert image.shape[-1] == 3, "Expected 3 channels, got %d" % image.shape[-1]
    image = _extract_object(image)

    obj_color_mean = _get_color_mean(image)
    obj_color_std = _get_color_std(image)
    # Resize về cùng một cỡ và đệm 1 vòng pixel 0 bên ngoài,
    # bỏ comment ở dưới sẽ thấy
    image = _pad_resize(image, shape)
    # Chồng hog và màu thành 1 vector

    feature = obj_color_mean, obj_color_std, _hog(image, shape)

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


def to_csv(path):
    train_folder = "data/train"
    shape = (256, 256)
    data = []

    for name in os.listdir(train_folder):
        img_path = os.path.join(train_folder, name)
        img = cv2.imread(img_path)
        color_mean, color_std, hog = extract_features(img, shape, img_path)
        color_mean_output_path = os.path.join(path, 'color_mean_' + name.split('.')[0] + '.npy')
        color_std_output_path = os.path.join(path, 'color_out_' + name.split('.')[0] + '.npy')
        hog_output_path = os.path.join(path, 'hog_' + name.split('.')[0] + '.npy')

        np.save(color_mean_output_path, color_mean)
        np.save(color_std_output_path, color_std)
        np.save(hog_output_path, hog)

        # if cols is None:
        #     cols = ['name']
        #     cols += ['color_' + str(i) for i in range(color_mean.shape[0])]
        #     cols += ['hog_' + str(i) for i in range(hog.shape[0])]

        data.append([name, color_mean_output_path, color_std_output_path, hog_output_path])

    df = pd.DataFrame(data, columns=['name', 'color_mean', 'color_std', 'hog'])
    df.to_csv('data.csv')
