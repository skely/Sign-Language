import os
import h5py
import matplotlib.pyplot as plt
import numpy as np


def my_reshape(_data):
    data_shape = np.shape(_data)
    new_channel_number = 3
    res = np.zeros((data_shape[0] * int(data_shape[2] / new_channel_number), data_shape[1], new_channel_number))
    for i in range(np.size(_data, 0)):
        tmp = _data[i]
        for j in range(61):
            res[61 * i + j, :, 0:3] = tmp[:, j * 3:j * 3 + 3]
    return res


if __name__ == '__main__':
    path = '/home/jedle/data/Sign-Language/_source_clean/'
    data_file = 'prepared_data_ang_30-30_aug20.h5'
    out_h5_file = 'aug20.h5'

    f = h5py.File(os.path.join(path, data_file), 'r')

    train_X = f['train_X']
    train_Y = f['train_Y']
    test_X = f['test_X']
    test_Y = f['test_Y']

    print(type(train_X))
    data = train_X, train_Y
    print(type(data[0]))
    data_shape = np.shape(train_X)
    print(data_shape)

    X = np.concatenate((my_reshape(train_X), my_reshape(test_X)))
    Y = np.concatenate((my_reshape(train_Y), my_reshape(test_Y)))

    hf = h5py.File(os.path.join(path, out_h5_file), 'w')
    hf.create_dataset('X', data=X)
    hf.create_dataset('Y', data=Y)
    hf.close()