import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Conv1D/tests'
    out_path = '/home/jedle/Projects/Sign-Language/tests/Conv3D/tests'
    data_h5_file = 'prepared_data_ang_aug10.h5'
    out_h5_file = 'simple_aug10.h5'

    f = h5py.File(os.path.join(path, data_h5_file), 'r')

    test_X = f['test_X']
    test_Y = f['test_Y']
    train_X = f['train_X']
    train_Y = f['train_Y']

    train_X = np.concatenate((train_X, test_X))
    train_Y = np.concatenate((train_Y, test_Y))

    train_Y = np.reshape(train_Y, (-1, 97, 3), order='F')

    train_X = np.swapaxes(train_X, 1, 2)
    train_Y = np.swapaxes(train_Y, 1, 2)

    train_X = np.reshape(train_X, (-1, 97))
    train_Y = np.reshape(train_Y, (-1, 97))

    print(np.shape(train_X))
    print(np.shape(train_Y))

    diff_X = np.diff(train_X)
    data_count = np.size(diff_X, 0)
    all_diffs = np.zeros((data_count))
    for i in range(data_count):
        all_diffs[i] = np.sum(diff_X[i, :])

    thrs_pwr = np.arange(20)
    thrs = np.power(10.0, -thrs_pwr)
    bellows = np.zeros(20)

    for j, thr in enumerate(thrs):
        bellow_thr = 0
        for i in range(data_count):
            if all_diffs[i] < thr:
                bellows[j] += 1

    print(np.shape(train_X))
    print(np.shape(train_Y))

    thr = 0.001
    data_list_X = []
    data_list_Y = []
    for i in range(data_count):
        if all_diffs[i] > thr:
            data_list_X.append(train_X[i, :])
            data_list_Y.append(train_Y[i, :])
    data_X = np.asarray(data_list_X)
    data_Y = np.asarray(data_list_Y)

    # print(np.shape(data_X))
    # print(np.shape(data_Y))

    # for i, thr in enumerate(thrs):
    #     print(thr)
    #     print('{} / {}'.format(bellows[i], data_count))
    #     print('{}'.format(bellows[i] / data_count))
    #     print('')
    #
    # show_ids = np.random.randint(np.size(data_X, 0), size=10)
    # for i in show_ids:
    #     print(i)
    #     plt.figure()
    #     plt.plot(data_X[i, :])
    #     plt.plot(data_Y[i, :])
    # plt.show()

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
