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


def make_log():

    plot_model(model, to_file=os.path.join(path, 'model_{}_run_{}.png'.format(test_name, i)), show_shapes=True)

    lines_list = []
    lines_list.append('*******************************\n')
    lines_list.append('test name: {}\n'.format(test_name))

    lines_list.append('hidden_layers: {}\n'.format(h))
    lines_list.append('skips: {}\n'.format(skips))
    lines_list.append('model plot: {}\n'.format(test_name + '.png'))

    lines_list.append('loss function: {}\n'.format(loss))
    lines_list.append('optimizer_name: {}\n'.format(optimizer_name))
    lines_list.append('learning_rate: {}\n'.format(learning_rate))
    lines_list.append('momentum: {}\n'.format(momentum))
    lines_list.append('decay: {}\n'.format(decay))

    lines_list.append('epochs: {}\n'.format(epochs))
    lines_list.append('batch_size: {}\n'.format(batch_size))

    lines_list.append('mse: {}\n'.format(mse))
    lines_list.append('accuracy: {}\n'.format(accuracy))
    lines_list.append('start_time: {}\n'.format(start_time))
    lines_list.append('end_time: {}\n'.format(end_time))
    lines_list.append('duration: {}\n'.format(end_time-start_time))

    with open(os.path.join(path, '{}.txt'.format(test_name)), 'a+') as f:
        f.writelines(lines_list)


if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    data = prepare_data_file(data_file)
    data_shape = np.shape(data[0])

    test_name = 'dense_3layer'
    skips = 'no'
    hidden_layer_sizes = [[9], [27], [81], [248], [9], [27], [81], [248], [9], [27], [81], [248]]
    loss = 'mean_squared_error'
    optimizer_name = 'adam'
    learning_rate = 0.01
    momentum = [0.9, 0.99]
    decay = 1e-7

    epochs = 200
    batch_size = 100

    for i, h in enumerate(hidden_layer_sizes):
        start_time = datetime.datetime.now()
        optimizer = optimizer_definition(optimizer_name, learning_rate, momentum, decay)
        model = model_definition(h, data_shape, loss, optimizer, batch_size, skips)
        model, evaluation, history = training(model, data, epochs, batch_size)
        _, mse, accuracy = evaluation
        end_time = datetime.datetime.now()
        make_log()
        model.save(os.path.join(path, 'model_{}_run_{}.h5'.format(test_name, i)))
        with open(os.path.join(path, 'history_{}_run_{}.pkl'.format(test_name, i)), 'wb') as f:
            pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
