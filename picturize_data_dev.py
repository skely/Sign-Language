from lib import BVH
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def picturize_data(_data):
    dimension = np.ndim(_data)
    if dimension == 3:
        batch, time, features = np.shape(_data)
        _data = np.reshape(_data, (batch, time, -1, 3))
    else:
        print('data has not expected shape (batch, time, features)')
    return _data


def depicturize_data(_data):
    dimension = np.ndim(_data)
    if dimension == 4:
        batch, time, features, channels = np.shape(_data)
        _data = np.reshape(_data, (batch, time, features*channels))
    else:
        print('data has not expected shape (batch, time, features, channels)')
    return _data


def get_minmax_3D(_data):
    batch, time, features, channels = np.shape(_data)
    min_max = np.ones((2, 3))
    min_max[0, :] = np.inf
    min_max[1, :] = -np.inf

    for item in range(batch):
        for channel in range(channels):
            tmp_min = np.min(_data[item, :, :, channel])
            tmp_max = np.max(_data[item, :, :, channel])

            if tmp_min < min_max[0, channel]:
                min_max[0, channel] = tmp_min
            if tmp_max > min_max[1, channel]:
                min_max[1, channel] = tmp_max

    return min_max


def normalize_3D(_data):
    """
    normalize features in data [item, time, features, channels])
    :param _data:
    :param _minmax:
    :return:
    """
    dimension = np.ndim(_data)
    if dimension == 3:
        _data = picturize_data(_data)
        # batch, time, features = np.shape(_data)
        # _data = np.reshape(_data, (batch, time, -1, 3))

    _minmax = get_minmax_3D(_data)

    _new_data = _data.copy()
    for i in range(np.size(_data, 0)):
        for ch in range(np.size(_data, 3)):
            _new_data[i, :, :, ch] = (_data[i, :, :, ch] - _minmax[0, ch]) / (_minmax[1, ch] - _minmax[0, ch])
    if dimension == 3:
        _data = depicturize_data(_data)
        # _data = np.reshape(_data, (batch, time, features))
    return _new_data


def do_stuff(data_selected, mask=None):
    if mask is None:
        caption = 'ground truth'
        do_3D_plot = True
    else:
        caption = 'known data'
        do_3D_plot = False
    do_imshow = True

    data_selected = picturize_data(data_selected)
    normed = normalize_3D(data_selected)

    # plot 3D
    # ****************
    batch_item = 0
    frame = 0
    if do_3D_plot:
        fig = plt.figure()
        result = normed[batch_item, frame, :, :]
        # result = np.reshape(result, (-1, 3))
        ax = fig.add_subplot(111, projection='3d', title='{}, (frame: {})'.format(caption, frame))
        for i in range(np.size(result, 0)):
            ax.scatter(result[i, 0], result[i, 1], result[i, 2], color=[result[i, 0], result[i, 1], result[i, 2]])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # ax.set_xlim3d(min_max[0, 0], min_max[1, 0])
        # ax.set_ylim3d(min_max[0, 1], min_max[1, 1])
        # ax.set_zlim3d(min_max[0, 2], min_max[1, 2])
        ax.set_xlim3d(0, 1.1)
        ax.set_ylim3d(0, 1.1)
        ax.set_zlim3d(0, 1.1)
        ax.scatter(1, 0, 0, color=[1, 0, 0], label='X')
        ax.scatter(0, 1, 0, color=[0, 1, 0], label='Y')
        ax.scatter(0, 0, 1, color=[0, 0, 1], label='Z')
        # plt.show()
        ax.legend()
    if mask==30:
        normed[0, mask:-mask, :, :] = 0

    if do_imshow:
        for i in range(1):
            plt.figure()
            plt.title(caption)
            result = np.swapaxes(normed, 1, 2)[i, :, :, :]
            plt.imshow(result)

if __name__ == '__main__':
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    data = np.load(data_file)
    train_X = data['train_X']
    train_Y = data['train_Y']

    # np.shape(train_X)
    # train_X = picturize_data(train_X)
    # minimaxi = get_minmax_3D(train_X)
    # testik = normalize_3D(train_X, minimaxi)

    do_stuff(train_Y)
    do_stuff(train_X, mask=30)
    # do_stuff(train_X-train_Y)
    plt.show()