import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from lib import data_prep

def analysis(_data, threshold=20):
    for kee in _data:
        # print('{} : shape={}'.format(kee, np.shape(data_loaded[kee])))
        batch, time, channels = np.shape(_data[kee])
        significance_vector = np.zeros((batch, channels))
        significance_bool = np.zeros((batch, channels))

        batch = 3
        for b in range(batch):
            for c in range(channels):
                sys.stdout.write("\rBatch {}/{} in progress.".format(b, batch - 1))
                # difference = np.diff(_data[kee][b, :, c])
                max_difference = abs(np.min(_data[kee][b, :, c]) - np.max(_data[kee][b, :, c]))
                # print('batch: {}\nchannel: {}\ndiff: {}\n*****\n'.format(b, c, max_difference))
                significance_vector[b, c] = max_difference
                if max_difference > threshold:
                    significance_bool[b, c] = 1
        break
    sys.stdout.flush()
    flag = False
    for c in range(channels):

        if sum(significance_bool[:, c]) / batch == 1 or sum(significance_bool[:, c]) / batch == 0:
            # print('{}. channel works'.format(c))
            pass
        else:
            flag = True
            print('{}. channel dont fit'.format(c))
            print(significance_bool[:batch, c])
            print(significance_vector[:batch, c])

    if not flag:
        print('all channels are OK.')
    return significance_vector


def analysis2(_data_array):
    batch, time, channels = np.shape(_data_array)

    other_channels = []
    above = 10
    for b in range(channels):

        irrelevant_channels = []
        suspicious_channels = []
        for ch in range(channels):
            # sys.stdout.write("\rChannel {}/{} in progress.".format(ch,channels))

            minmax_diff = np.max(_data_array[b, :, ch]) - np.min(_data_array[b, :, ch])
            frame_diff = np.max(np.diff(_data_array[b, :, ch])) - np.min(np.diff(_data_array[b, :, ch]))
            # print('{:4f}, {:4f}'.format(minmax_diff, frame_diff))
            sys.stdout.flush()
            if minmax_diff < 0.0008:
                irrelevant_channels.append(ch)
            elif minmax_diff < 0.01:
                sys.stdout.write("\rBatch {}/{} in progress.\n".format(b, batch))
                suspicious_channels.append(ch)
                print('minmax diff: {}, frame_diff{}'.format(minmax_diff, frame_diff))
            else:
                if minmax_diff < above:
                    above = minmax_diff
                    dif = frame_diff

        # print('irrelevant channels: {}/{}'.format(len(irrelevant_channels), channels))
        # print('suspicious channels: {} ({})'.format(len(suspicious_channels), suspicious_channels))
    print('above min = {} (framedif: {})'.format(above, dif))

if __name__ == '__main__':
    source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
    prepared_data_file = os.path.join(source_dir, 'prepared_data_30-30_aug10times2.npz')
    data_loaded = np.load(prepared_data_file)

    data_array = data_loaded['train_Y']
    batch, time, features = np.shape(data_array)
    print(batch, time, features)
    for i in range(batch):
        print(np.average(data_array[i, :, 0]))

        # plt.figure()
        # plt.plot(data_array[i, :, 0])
        # plt.figure()
        # plt.plot(data_array[i, :, 1])
        # plt.figure()
        # plt.plot(data_array[i, :, 2])
        # plt.show()