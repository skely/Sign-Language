import os
import numpy as np
import h5py

if __name__ == '__main__':
    # path = 'data'
    path = '/home/jedle/data/Sign-Language/_source_clean/'
    out_path = '/home/jedle/Projects/Sign-Language/nn_tests/data'
    data_h5_file = 'prepared_data_ang_30-30_aug15.h5'
    out_h5_file = '3D_aug15.h5'

    f = h5py.File(os.path.join(path, data_h5_file), 'r')
    # f = np.load(os.path.join(path, data_h5_file))

    test_X = f['test_X']
    test_Y = f['test_Y']
    train_X = f['train_X']
    train_Y = f['train_Y']

    data_X = np.concatenate((train_X, test_X))
    data_Y = np.concatenate((train_Y, test_Y))

    print(np.shape(train_X))
    print(np.shape(train_Y))

    data_length = np.size(data_X, 0)
    split = 0.1
    split_point = int((1-split) * data_length)

    train_X = data_X[:split_point, :]
    test_X = data_X[split_point:, :]

    train_Y = data_Y[:split_point, :]
    test_Y = data_Y[split_point:, :]

    hf = h5py.File(os.path.join(out_path, out_h5_file), 'w')
    hf.create_dataset('train_X', data=train_X)
    hf.create_dataset('train_Y', data=train_Y)
    hf.create_dataset('test_X', data=test_X)
    hf.create_dataset('test_Y', data=test_Y)
    hf.close()
