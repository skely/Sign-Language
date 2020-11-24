import sys
import argparse
import os
import datetime
import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Dense, Input, Conv1D, Flatten, concatenate, LSTM
from keras.models import Model, load_model
from keras.optimizers import sgd, adam
from contextlib import redirect_stdout
import random
import string

def get_random_alphanumeric_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    # print("Random alphanumeric String is:", result_str)
    return result_str


def call_optimizer(_opt):
    if _opt == 'adam':
        _optimizer = adam(learning_rate=0.01, beta_1=0.9 ,beta_2=0.999, epsilon=1e-07, amsgrad=False)
    elif _opt == 'sgd':
        _optimizer = sgd(lr=lr, momentum=momentum, decay=decay)
    else:
        print('invalid optimizer: {}'.format(_opt))
        _optimizer = None
    return _optimizer


def define_model_conv():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(input)
    layer2 = Conv1D(filters=16, kernel_size=3, activation=_activation, padding='same')(layer1)
    layer3 = Conv1D(filters=32, kernel_size=3, activation=_activation, padding='same')(layer2)
    layer5 = Flatten()(layer3)
    layer6 = Dense(97*3, activation=_activation)(layer5)

    _model = Model(inputs=input, outputs=layer6, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def define_model_dilconv():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = Conv1D(filters=9, kernel_size=3, activation=_activation, dilation_rate=2, padding='same')(input)
    layer2 = Conv1D(filters=9, kernel_size=9, activation=_activation, dilation_rate=2, padding='same')(layer1)
    layer3 = Conv1D(filters=9, kernel_size=27, activation=_activation, dilation_rate=2, padding='same')(layer2)
    layer4 = Conv1D(filters=9, kernel_size=45, activation=_activation, dilation_rate=2, padding='same')(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(97*3, activation=_activation)(layer5)

    _model = Model(inputs=input, outputs=layer6, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def define_model():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(input)
    concat1 = concatenate([input, layer1])
    layer2 = Conv1D(filters=16, kernel_size=3, activation=_activation, padding='same')(concat1)
    concat2 = concatenate([concat1, layer2])
    layer3 = Conv1D(filters=32, kernel_size=3, activation=_activation, padding='same')(concat2)
    concat3 = concatenate([concat2, layer3])
    layer4 = Conv1D(filters=64, kernel_size=3, activation=_activation, padding='same')(concat3)
    concat4 = concatenate([concat3, layer4])
    layer5 = Flatten()(concat4)
    layer6 = Dense(97*3, activation=_activation)(layer5)

    _model = Model(inputs=input, outputs=layer6, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def define_model_flat():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(input)
    concat1 = concatenate([input, layer1])
    layer2 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(concat1)
    concat2 = concatenate([concat1, layer2])
    layer3 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(concat2)
    concat3 = concatenate([concat2, layer3])
    layer4 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(concat3)
    concat4 = concatenate([concat3, layer4])
    layer5 = Flatten()(concat4)
    layer6 = Dense(97*3, activation=_activation)(layer5)

    _model = Model(inputs=input, outputs=layer6, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def define_model_short():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = Conv1D(filters=8, kernel_size=3, activation=_activation, padding='same')(input)
    concat1 = concatenate([input, layer1])
    layer2 = Conv1D(filters=16, kernel_size=5, activation=_activation, padding='same')(concat1)
    concat2 = concatenate([concat1, layer2])
    layer_flatten = Flatten()(concat2)
    output = Dense(97*3, activation=_activation)(layer_flatten)

    _model = Model(inputs=input, outputs=output, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def define_model_lstm():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = LSTM(64, activation=_activation, return_sequences=True)(input)
    layer2 = LSTM(16, activation=_activation, return_sequences=True)(layer1)
    layer3 = LSTM(3, activation=_activation, return_sequences=True)(layer2)
    # layer_flatten = Flatten()(layer2)
    # output = Dense(97*3, activation=_activation)(layer_flatten)

    _model = Model(inputs=input, outputs=layer3, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def define_model_lstm_v2():
    _loss = 'mean_squared_error'
    _optimizer = call_optimizer(opt)
    _activation = 'sigmoid'

    input = Input(shape=(97, 3))
    layer1 = LSTM(64, activation=_activation, return_sequences=True)(input)
    layer2 = LSTM(3, activation=_activation, return_sequences=True)(layer1)
    # layer_flatten = Flatten()(layer2)
    # output = Dense(97*3, activation=_activation)(layer_flatten)

    _model = Model(inputs=input, outputs=layer2, name=test_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], validation_data=(_data[2], _data[3]), epochs=_epochs, batch_size=_batch_size, verbose=2)
    _evaluations = _model.evaluate(_data[2], _data[3])
    print('loss: {}, mse: {}'.format(_evaluations[0], _evaluations[1]))
    return _model, _evaluations, _history


def log():
    # plot_model(model, to_file=os.path.join(path, test_name), show_shapes=True)     # saves figure of model
    model.save(os.path.join(path, 'model_{}.h5'.format(test_name)))                # saves model hdf5
    with open(os.path.join(path, 'history_{}.pkl'.format(test_name)), 'wb') as f:  # saves training history
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(path, 'model_summary_{}.txt'.format(test_name)), 'w') as f:  # saves model summary to the file
        with redirect_stdout(f):
            model.summary()

    lines_list = []
    lines_list.append('*******************************\n')
    lines_list.append('test name: {}\n'.format(test_name))
    lines_list.append('model file name: model_{}.h5\n'.format(test_name))
    if 'loaded_model' in globals():
        lines_list.append('loaded model file name (continuous training): {}\n'.format(loaded_model))
    else:
        lines_list.append('loaded model file name (continuous training): None (zero generation)}\n')
    lines_list.append('training history: history_{}.pkl\n'.format(test_name))
    lines_list.append('model_visualization: {}.png\n'.format(test_name))
    lines_list.append('training dataset: {}\n'.format(data_file))
    lines_list.append('epochs: {}\n'.format(epochs))
    lines_list.append('batch: {}\n'.format(batch))
    lines_list.append('loss: {}\n'.format(evaluation[0]))
    lines_list.append('mse: {}\n'.format(evaluation[1]))
    lines_list.append('optimizer: {}\n'.format(opt))
    lines_list.append('learning_rate: {}\n'.format(lr))
    lines_list.append('momentum: {}\n'.format(momentum))
    if 'decay' in globals():
        lines_list.append('decay: {}\n'.format(decay))
    lines_list.append('elapsed time: {}\n'.format(end_time_stamp - time_stamp))

    with open(os.path.join(path, 'all_logs.txt'.format(test_name)), 'a') as f:
        f.writelines(lines_list)


def train_test_split(_data, _p=0.1, _flatten=False):
    data_items = np.size(_data, 0)
    train = _data[:int(data_items*(1-_p))]
    test = _data[int(data_items*(1-_p)):]
    if _flatten:
        test = np.reshape(test, (-1, 97*3))
    return train, test


def load_data(_file_name, _version='norm', _load_size=100):
    f = h5py.File(_file_name, 'r')

    if _version == 'norm':
        X = f['X'][:_load_size]
        Y = f['Y'][:_load_size]
        return X, Y

    elif _version == 'flip':
        X = f['X'][:_load_size]
        Y = f['Y'][:_load_size]
        X_rev = np.flip(X, axis=1)
        Y_rev = np.flip(Y, axis=1)
        return np.concatenate((X, X_rev)), np.concatenate((Y, Y_rev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', help='learning rate (default lr=0.01)', required=True, type=float)
    parser.add_argument('-m', '--loaded_model', help='model path (if continuous training) or model_type if creating new model (conv2l, conv4l, etc.)', type=str)
    args = parser.parse_args()

    # path = '/home/jedle/Projects/Sign-Language/nn_tests/data'
    path = '/storage/plzen1/home/jedlicka/Sign-Language/nn_tests/data'
    data_file = 'aug20.h5'
    # model_type = 'conv2l'
    # loaded_model = 'model_conv2l_gen0_20-09-29-22-26.h5'
    if 'model' in args.loaded_model:
        loaded_model = args.loaded_model
        loaded_generation = int(loaded_model.split('_')[2][3:])
        generation = 'gen{}'.format(loaded_generation+1)
        model_type = loaded_model.split('_')[1]
    else:
        model_type = args.loaded_model
        generation = 'gen0'

    time_stamp = datetime.datetime.now()
    # time_string = '{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(time_stamp.year%100, time_stamp.month, time_stamp.day, time_stamp.hour, time_stamp.minute)
    time_string = '{:02d}-{:02d}-{:02d}-{}'.format(time_stamp.year%100, time_stamp.month, time_stamp.day, get_random_alphanumeric_string(5))
    test_name = '{}_{}_{}'.format(model_type, generation, time_string)

    limited_batch_size = 100000  #  aug20 length = 1715320
    epochs = 200
    batch = 200
    lr = args.learning_rate
    momentum = 0
    opt = 'adam'
    # opt = sgd(lr=lr, momentum=momentum, decay=decay)
    # optimizer = adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

    decay = lr / epochs

    # f = h5py.File(os.path.join(path, data_file), 'r')
    # listkeys = f.keys()
    # X = f['X'][:limited_batch_size]
    # Y = f['Y'][:limited_batch_size]

    X, Y = load_data(os.path.join(path, data_file), _version='flip', _load_size=limited_batch_size)

    train_X, test_X = train_test_split(np.array(X))

    if model_type in ['lstm1l', 'lstm2l']:
        flatten = False
    else:
        flatten = True
    train_Y, test_Y = train_test_split(np.array(Y), _flatten=flatten)
    
    data = train_X, train_Y, test_X, test_Y

    if 'loaded_model' in globals():
        model = load_model(os.path.join(path, loaded_model))
        K.set_value(model.optimizer.lr, lr)
    elif model_type == 'conv2l':
        model = define_model_short()
    elif model_type == 'lstmV1':
        model = define_model_lstm()
    elif model_type == 'conv4l':
        model = define_model_flat()
    elif model_type == 'c4lexp':
        model = define_model()
    elif model_type == 'c3lconv':
        model = define_model_conv()
    elif model_type == 'dil4l':
        model = define_model_dilconv()
    elif model_type == 'lstmV2':
        model = define_model_lstm_v2()
    else:
        print('Unrecognized model type')

    model, evaluation, history = training(model, data, epochs, batch)
    end_time_stamp = datetime.datetime.now()
    log()
