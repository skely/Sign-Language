import os
import datetime
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import adam, sgd
from keras.utils import plot_model


def data_normalization(_data):
    _min = np.min(_data)
    _max = np.max(_data)

    _normed_data = (_data - _min) / (_max - _min)
    return _normed_data, [_min, _max]


def my_resize_dense_3d(_data):
    new_feat = 3
    batch, time, features = np.shape(_data)
    _new_data = np.zeros((batch * int(features / new_feat), time, new_feat))

    for b in range(batch):
        for f in range(int(features / new_feat)):
            _new_data[b * f:(b + 1) * f, :, :] = _data[b, :, f * 3:(f + 1) * 3]
    return _new_data


def visual_testing(_test_X, _test_Y, _model, _item):
    _response = _model.predict(_test_X)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.title('axis: {}'.format(i))
        plt.plot(_response[_item, :, i], label='predict')
        plt.plot(_test_X[_item, :, i], label='input')
        plt.plot(_test_Y[_item, :, i], label='orig')
    plt.legend()
    plt.show()


def prepare_data_file(_data_file, normalize=True, shuffle=False, resize='3d'):
    _data = np.load(_data_file)
    _train_X = _data['train_X']
    _train_Y = _data['train_Y']
    _test_X = _data['test_X']
    _test_Y = _data['test_Y']
    if normalize:
        _train_X, minmax = data_normalization(_train_X)
        _train_Y, minmax = data_normalization(_train_Y)
        _test_X, minmax = data_normalization(_test_X)
        _test_Y, minmax = data_normalization(_test_Y)
    if resize == '3d':
        _train_X = my_resize_dense_3d(_train_X)
        _train_Y = my_resize_dense_3d(_train_Y)
        _test_X = my_resize_dense_3d(_test_X)
        _test_Y = my_resize_dense_3d(_test_Y)
    if shuffle:
        # messes with dimensions (one added !)
        random.seed(9)
        pattern = random.shuffle(list(range(np.size(_train_X, 0))))
        _train_X = _train_X[pattern, :, :]
        _train_Y = _train_Y[pattern, :, :]
        _test_X = _test_X[pattern, :, :]
        _test_Y = _test_Y[pattern, :, :]
        print(np.shape(_train_X))

    return _train_X, _train_Y, _test_X, _test_Y


def optimizer_definition(_optimizer_name, _learning_rate, _momentum, _decay):

    if _optimizer_name == 'adam':
        _m1, _m2 = _momentum
        _opt = adam(learning_rate=_learning_rate, beta_1=_m1, beta_2=_m2, epsilon=_decay)
    elif _optimizer_name == 'sgd':
        _opt = sgd(lr=_learning_rate, momentum=_momentum, decay=_decay, nesterov=False)

    return _opt


def model_definition(_layers, _data_shape, _loss, _optimizer, _batch_size, _skips):
    _activation_function = 'relu'
    batch, time, features = _data_shape
    depth = len(_layers)

    input_layer = Input(shape=(time, features))
    last_layer = input_layer
    for i in range(depth):
        new_layer = Dense(_layers[i], activation=_activation_function)(last_layer)
        if _skips == 'no':
            last_layer = new_layer
        elif _skips == 'simple':
            skip_layer = concatenate([new_layer, last_layer])
            last_layer = skip_layer

    output_layer = Dense(features, activation=_activation_function)(last_layer)
    _model = Model(inputs=input_layer, outputs=output_layer, name='model_from_definition')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error', 'accuracy'])
    _model.summary()
    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _evaluations = _model.evaluate(_data[2], _data[3])
    return _model, _evaluations, _history


def prediction(_model, test_input):
    return _model.predict(test_input)


def compare(_orig, _comp):
    mse = ((_orig - _comp) ** 2).mean(axis=1)
    return np.mean(mse)


def run_model_predict(_path, _data_file, _model_file, _batch, _channel, train_data=False):
    train_X, train_Y, test_X, test_Y = prepare_data_file(_data_file)
    if train_data:
        test_Y = train_Y
        test_X = train_X

    model = load_model(_model_file)
    res = model.predict(test_X)

    print('test Y vs test X')
    print(compare(test_Y[_batch, :, :], test_X[_batch, :, :]))
    print('test Y vs prediction')
    print(compare(test_Y[_batch, :, :], res[_batch, :, :]))

    plt.figure()
    plt.title('axis: {}'.format(['X', 'Y', 'Z'][_channel]))
    # plt.plot(res[_batch, :, _channel], label='prediction')
    plt.plot(test_Y[_batch, :, _channel], label='ground truth')
    plt.plot(test_X[_batch, :, _channel], label='cubic interpolation (input)')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(_path, 'figure-{}-{}-data'.format(_batch, _channel)))
    return res

if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    model_file = '/home/jedle/Projects/Sign-Language/tests/Dense/tests/model_dense_3layer_sgd_lr01_skips_bias_run_9.h5'

    batch = 0
    channel = 0

    run_model_predict(path, data_file, model_file, batch, channel)

    # train_X, train_Y, test_X, test_Y = prepare_data_file(data_file)
    # model = load_model(model_file)
    # res = model.predict(test_X)
    #
    # print('test Y vs test X')
    # print(compare(test_Y[batch, :, :], test_X[batch, :, :]))
    # print('test Y vs prediction')
    # print(compare(test_Y[batch, :, :], res[batch, :, :]))
    #
    # plt.plot(res[batch, :, channel], label='prediction')
    # plt.plot(test_Y[batch, :, channel], label='groung truth')
    # plt.plot(test_X[batch, :, channel], label='cubic interpolation')
    # plt.legend()
    # plt.show()