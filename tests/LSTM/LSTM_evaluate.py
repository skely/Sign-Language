import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from lib import data_prep
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler


def read_logfile(_log_file):
    with open(_log_file, 'r') as jf:
        _log = json.load(jf)

    return _log


def sort_results_by_loss(_log):
    kernels = []
    epochs = []
    loss = []
    val_loss = []
    for l in log:
        # print(l)
        kernels.append(l['kernels'])
        epochs.append(l['epochs'])
        loss.append(l['loss'])
        val_loss.append(l['val_loss'])

    order = np.argsort(loss)

    print('{:<10} {:<10} {:<10} {:<10}'.format('kernels', 'epochs', 'loss', 'val_loss'))
    for i in order:
        print('{:<10} {:<10} {:<10.5f} {:<10.5f}'.format(kernels[i], epochs[i], loss[i], val_loss[i]))
    return kernels[order[0]], epochs[order[0]]

if __name__ == '__main__':
    work_dir = '/home/jedle/data/Sign-Language/_source_clean/testing/test_glo_v2/'
    # work_dir = '/home/jedle/data/Sign-Language/_source_clean/testing/test_v4/'
    log_file = os.path.join(work_dir, 'losses.txt')
    log = read_logfile(log_file)
    top = sort_results_by_loss(log)

    model_file = os.path.join(work_dir, 'model_LSTM_AE_ker{}_ep{}.h5'.format(top[0], top[1]))
    # model_file = os.path.join(work_dir, 'model_LSTM_AE_ker200_ep1000.h5')
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    model = load_model(model_file)
    data = np.load(data_file)
    train_Y = data['train_Y']
    test_Y = data['test_Y']
    train_X = data['train_X']
    test_X = data['test_X']

    prediction = Model.predict(model, train_Y)

    # print(np.shape(train_Y))
    # print(np.shape(test_Y))
    # print(np.shape(train_X))
    # print(np.shape(test_X))
    # print(np.shape(prediction))

    plt.plot(train_X[10, :, 22], label='train_X')
    plt.plot(train_Y[10, :, 22], label='train_Y')
    plt.plot(prediction[10, :, 22], label='NN prediction')
    plt.show()

    list_lin = []
    list_neu = []
    for tmp_sign in range(np.size(prediction, 0)):
        linear_comp = data_prep.sign_comparison(train_X[tmp_sign, :, :], train_Y[tmp_sign, :, :])
        neural_comp = data_prep.sign_comparison(train_X[tmp_sign, :, :], prediction[tmp_sign, :, :])
        list_lin.append(linear_comp)
        list_neu.append(neural_comp)
        sys.stdout.write('\rprocessing... {:.2f}% done.'.format(100 * (tmp_sign + 1) / np.size(prediction, 0)))
    sys.stdout.write('\rdone.\n')

    print('linear: {}\nneural: {}'.format(np.average(list_lin), np.average(list_neu)))
