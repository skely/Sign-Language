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
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import fastdtw
import json


def train_LSTM_bidirectional(_datafile, _output_dir, _LSTM_kernels, _epochs):
    """
    Trains LSTM bidirectional network with given parameters.
    :param _datafile: datafile prepared by LSTM_dataprep
    :param _output_dir: model and log is saved there
    :param _LSTM_kernels:
    :param _epochs:
    :return:
    """
    model_code = 'model_BI_AE_lstm{}_ep{}'.format(_LSTM_kernels, _epochs)
    model_file = os.path.join(_output_dir, model_code + '.h5')
    log_file = os.path.join(_output_dir, model_code + '.txt')

    data = np.load(_datafile)
    train_X = data['train_X']
    train_Y = data['train_Y']

    batch, time, features = np.shape(train_X)

    model = Sequential()
    model.add(Bidirectional(LSTM(_LSTM_kernels, activation='relu'), input_shape=(time, features)))
    model.add(RepeatVector(time))
    model.add(Bidirectional(LSTM(_LSTM_kernels, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(features)))
    # model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='SGD', loss='mse')
    print(model.summary())
    history = model.fit(train_X, train_Y, epochs=_epochs, validation_split=0.1, shuffle=True, verbose=1, batch_size=100)
    model.save(model_file)

    with open(log_file, 'w') as f:
        json.dump(history.history, f)

    return history

if __name__ == '__main__':
    results = []
    set_epochs = [500]
    set_kernels = [200, 250, 300, 350, 400]

    for k in set_kernels:
        for e in set_epochs:
            LSTM_kernels = k
            epochs = e
            source_dir = '/home/jedle/data/Sign-Language/_source_clean/testing'
            prepared_data_file = os.path.join(source_dir, 'prepared_data.npz')

            history = train_LSTM_bidirectional(prepared_data_file, source_dir, LSTM_kernels, epochs)
            results.append('Kernels: {}, epochs: {}, loss: {:.4f}, val_loss: {:.4f}\n'.format(LSTM_kernels, epochs, history.history['loss'][-1], history.history['val_loss'][-1]))

    for res in results:
        print(res)

    with open(os.path.join(source_dir, 'losses.txt')) as f:
        f.write(results)