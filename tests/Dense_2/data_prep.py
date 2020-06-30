import os
import numpy as np
import random
import matplotlib.pyplot as plt


def data_reshape(_data, _mode='oneax'):
    train_X, train_Y, test_X, test_Y = _data

    if _mode == 'orig':
        train_X = data_reshape_array(train_X)
        train_Y = data_reshape_array(train_Y)
        test_X = data_reshape_array(test_X)
        test_Y = data_reshape_array(test_Y)
    if _mode == 'oneax':
        train_X = data_reshape_oneaxis(train_X)
        train_Y = data_reshape_oneaxis(train_Y)
        test_X = data_reshape_oneaxis(test_X)
        test_Y = data_reshape_oneaxis(test_Y)

    return  train_X, train_Y, test_X, test_Y


def data_reshape_oneaxis(_data):
    selected_channel = 1
    batch, time, channels = np.shape(_data)
    reshaped_data = np.zeros((int(batch * channels / 3), time))
    # print(np.shape(reshaped_data))
    for i in range(batch):
        for j in range(int(channels/3)):
            reshaped_data[i*int(channels/3)+j, :] = np.transpose(_data[i, :, j*3+selected_channel])


    return reshaped_data


def data_reshape_array(_data):
    batch, time, channels = np.shape(_data)
    reshaped_data = np.zeros((batch*channels, time))
    for i in range(batch):
        reshaped_data[i*channels:(i+1)*channels, :] = np.transpose(_data[i, :, :])
    return reshaped_data


def read(data_file):
    data = np.load(data_file)
    train_X = data['train_X']
    train_Y = data['train_Y']
    test_X = data['test_X']
    test_Y = data['test_Y']
    return train_X, train_Y, test_X, test_Y


def data_normalization_get_min_max(_data):
    _min = np.min(_data)
    _max = np.max(_data)
    return _min, _max


def data_normalization(_data, _minmax=None):
    if _minmax == None:
        _min, _max = data_normalization_get_min_max(_data)
    else:
        _min, _max = _minmax

    _normed_data = (_data - _min) / (_max - _min)
    return _normed_data, [_min, _max]


def normalization(_data):
    _data_concat = np.concatenate(_data)
    _data_concat_normed, _ = data_normalization(_data_concat)
    sizes = []
    for i in range(len(_data)):
        if i == 0:
            sizes.append(np.shape(_data[i])[0])
        else:
            sizes.append((np.shape(_data[i])[0]) + sizes[len(sizes) - 1])

    train_X = _data_concat_normed[0:sizes[0], :]
    train_Y = _data_concat_normed[sizes[0]:sizes[1], :]
    test_X = _data_concat_normed[sizes[1]:sizes[2], :]
    test_Y = _data_concat_normed[sizes[2]:, :]
    return train_X, train_Y, test_X, test_Y


def shuffle(_data, _seed=9):
    # messes with dimensions (one added !)
    _train_X, _train_Y, _test_X, _test_Y = _data
    random.seed(_seed)
    pattern = random.shuffle(list(range(np.size(_train_X, 0))))
    _train_X = _train_X[pattern, :, :]
    _train_Y = _train_Y[pattern, :, :]
    # pattern = random.shuffle(list(range(np.size(_test_X, 0))))
    _test_X = _test_X[pattern, :, :]
    _test_Y = _test_Y[pattern, :, :]
    # print(np.shape(_train_X))
    # print(np.shape(_test_X))
    return _train_X[0], _train_Y[0], _test_X[0], _test_Y[0]


def main(data_file):
    data = read(data_file)
    data = data_reshape(data)
    data = normalization(data)
    data = shuffle(data)
    return data


if __name__ == '__main__':
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    data = main(data_file)

