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

def data_normalization_get_min_max(_data):
    _min = np.min(_data)
    _max = np.max(_data)
    return _min, _max

def data_normalization(_data):
    _min, _max = data_normalization_get_min_max(_data)
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


def load_data_file(_data_file):
    _data = np.load(_data_file)
    _train_X = _data['train_X']
    _train_Y = _data['train_Y']
    _test_X = _data['test_X']
    _test_Y = _data['test_Y']
    return _train_X, _train_Y, _test_X, _test_Y


def prepare_data_file(_data_file, normalize=True, shuffle=False, resize='3d', reduce=True):

    _data = load_data_file(_data_file)
    min_max_all = np.zeros((4, 2))
    if normalize:
        _data_concat = np.concatenate(_data)
        _data_concat_normed, _ = data_normalization(_data_concat)
        sizes = []
        for i in range(len(_data)):
            if i == 0 :
                sizes.append(np.shape(_data[i])[0])
            else:
                sizes.append((np.shape(_data[i])[0]) + sizes[len(sizes) - 1])

        _train_X = _data_concat_normed[0:sizes[0], :, :]
        _train_Y = _data_concat_normed[sizes[0]:sizes[1], :, :]
        _test_X = _data_concat_normed[sizes[1]:sizes[2], :, :]
        _test_Y = _data_concat_normed[sizes[2]:, :, :]

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

    if reduce:
        print(np.shape(_train_X))
        _train_X = _train_X[:, :, 0]
        _train_Y = _train_Y[:, :, 0]
        _test_X = _test_X[:, :, 0]
        _test_Y = _test_Y[:, :, 0]

    return _train_X, _train_Y, _test_X, _test_Y, min_max_all


def optimizer_definition(_optimizer_name, _learning_rate, _momentum, _decay):

    if _optimizer_name == 'adam':
        _m1, _m2 = _momentum
        _opt = adam(learning_rate=_learning_rate, beta_1=_m1, beta_2=_m2, epsilon=_decay)
    elif _optimizer_name == 'sgd':
        _opt = sgd(lr=_learning_rate, momentum=_momentum, decay=_decay, nesterov=False)

    return _opt


def model_definition_1d(_layers, _data_shape, _loss, _optimizer, _batch_size, _skips, _activation_function):
    # _activation_function = 'relu'
    batch, time = _data_shape
    depth = len(_layers)

    input_layer = Input(shape=(time,))
    last_layer = input_layer
    for i in range(depth):
        new_layer = Dense(_layers[i], activation=_activation_function[i], use_bias=True)(last_layer)
        if _skips == 'no':
            last_layer = new_layer
        elif _skips == 'simple':
            skip_layer = concatenate([new_layer, last_layer])
            last_layer = skip_layer
    output_layer = Dense(time, activation=_activation_function[-1], use_bias=True)(last_layer)

    _model = Model(inputs=input_layer, outputs=output_layer, name='model_from_definition')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error', 'accuracy'])
    _model.summary()
    return _model


def model_definition(_layers, _data_shape, _loss, _optimizer, _batch_size, _skips, _activation_function):
    # _activation_function = 'relu'
    batch, time, features = _data_shape
    depth = len(_layers)

    input_layer = Input(shape=(time, features))
    last_layer = input_layer
    for i in range(depth):
        new_layer = Dense(_layers[i], activation=_activation_function[i], use_bias=True)(last_layer)
        if _skips == 'no':
            last_layer = new_layer
        elif _skips == 'simple':
            skip_layer = concatenate([new_layer, last_layer])
            last_layer = skip_layer
    output_layer = Dense(features, activation=_activation_function[-1], use_bias=True)(last_layer)

    _model = Model(inputs=input_layer, outputs=output_layer, name='model_from_definition')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error', 'accuracy'])
    _model.summary()
    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=2)
    _evaluations = _model.evaluate(_data[2], _data[3])
    return _model, _evaluations, _history


def prediction(_model, test_input):
    return _model.predict(test_input)


def make_log():
    model_plot_name = 'model_{}_run_{}.png'.format(test_name, i)
    model_name = 'model_{}_run_{}.h5'.format(test_name, i)

    plot_model(model, to_file=os.path.join(path, model_plot_name), show_shapes=True)

    lines_list = []
    lines_list.append('*******************************\n')
    lines_list.append('test name: {}\n'.format(test_name))
    lines_list.append('model file name: {}\n'.format(model_name))

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

    with open(os.path.join(path, 'results_{}.txt'.format(test_name)), 'a+') as f:
        f.writelines(lines_list)


if __name__ == '__main__':
    path = '/tests/old/Dense/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    data = prepare_data_file(data_file)
    data_shape = np.shape(data[0])
    print(data_shape)

    test_name = 'dense_4layer_v2_tst2'
    # skips = 'simple'
    skips = 'no'

    epochs = 1000
    batch_size = 100
    hidden_layer_sizes = [[9, 9], [27, 27], [81, 81], [248, 248]]
    activation_function = ['sigmoid', 'sigmoid', 'sigmoid']
    loss = 'mean_squared_error'
    optimizer_name = 'sgd'
    learning_rate = 0.1
    # momentum = [0.9, 0.99]
    momentum = 0.99
    decay = 1e-8/epochs


    for i, h in enumerate(hidden_layer_sizes):
        start_time = datetime.datetime.now()
        optimizer = optimizer_definition(optimizer_name, learning_rate, momentum, decay)
        model = model_definition_1d(h, data_shape, loss, optimizer, batch_size, skips, activation_function)
        model, evaluation, history = training(model, data, epochs, batch_size)
        _, mse, accuracy = evaluation
        end_time = datetime.datetime.now()
        make_log()
        model.save(os.path.join(path, 'model_{}_run_{}.h5'.format(test_name, i)))
        with open(os.path.join(path, 'history_{}_run_{}.pkl'.format(test_name, i)), 'wb') as f:
            pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
