import os
import h5py
import numpy as np


def check_sizes(fname):
    h5file = h5py.File(os.path.join(path, fname), 'r')

    test_X = h5file['test_X'][:]
    test_Y = h5file['test_Y'][:]
    train_X = h5file['train_X'][:]
    train_Y = h5file['train_Y'][:]
    print(fname)
    print(np.shape(test_X))
    print(np.shape(train_X))
    print(np.shape(test_Y))
    print(np.shape(train_Y))
    print()
    h5file.close()

    return train_X, train_Y, test_X, test_Y

path = '/home/jedle/Projects/Sign-Language/nn_tests/data'
fname10 = '3D_aug10.h5'
fname15 = '3D_aug15.h5'

data_10 = check_sizes(fname10)
data_15 = check_sizes(fname15)

print(type(data_15))
