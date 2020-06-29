import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
# import tests.Dense_2.data_prep as data_prep
import data_prep
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import sgd


def define_model():
    _model_name = 'Dense_12layer'
    _loss = 'mean_squared_error'
    _optimizer = 'sgd'
    lr = 0.1
    momentum = 0.
    decay = 1e-8/epochs
    _optimizer = sgd(lr, momentum, decay=decay)
    _activation = 'sigmoid'

    inputs = Input(shape=(97,))
    layer_1 = Dense(97, activation=_activation, use_bias=True)(inputs)
    layer_2 = Dense(97*97, activation=_activation, use_bias=True)(layer_1)
    layer_3 = Dense(97*97, activation=_activation, use_bias=True)(layer_2)
    layer_4 = Dense(97*97, activation=_activation, use_bias=True)(layer_3)
    # layer_4 = Dense(97*3*3*3*3, activation=_activation, use_bias=True)(layer_3)
    # layer_5 = Dense(97*3*3*3*3*3, activation=_activation, use_bias=True)(layer_4)
    # layer_6 = Dense(97*3*3*3*3*3*3, activation=_activation, use_bias=True)(layer_5)
    # layer_7 = Dense(97*3*3*3*3*3*3, activation=_activation, use_bias=True)(layer_6)
    # layer_8 = Dense(97*3*3*3*3*3*3, activation=_activation, use_bias=True)(layer_7)
    # layer_9 = Dense(97*3*3*3*3*3*3, activation=_activation, use_bias=True)(layer_8)
    # layer_10 = Dense(97*3*3*3*3*3*3, activation=_activation, use_bias=True)(layer_9)
    # layer_11 = Dense(97*3*3*3*3*3*3, activation=_activation, use_bias=True)(layer_10)
    layer_12 = Dense(97, activation=_activation, use_bias=True)(layer_4)

    _model = Model(inputs=inputs, outputs=layer_12, name=_model_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=2)
    _evaluations = _model.evaluate(_data[2], _data[3])
    print('loss: {}, mse: {}'.format(_evaluations[0], _evaluations[1]))
    return _model, _evaluations, _history


def log():
    plot_model(model, to_file=os.path.join(path, test_name), show_shapes=True)     # saves figure of model
    model.save(os.path.join(path, 'model_{}.h5'.format(test_name)))                # saves model hdf5
    with open(os.path.join(path, 'history_{}.pkl'.format(test_name)), 'wb') as f:  # saves training history
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

    lines_list = []
    lines_list.append('*******************************\n')
    lines_list.append('test name: {}\n'.format(test_name))
    lines_list.append('model file name: model_{}.h5\n'.format(test_name))
    lines_list.append('training history: history_{}.pkl\n'.format(test_name))
    lines_list.append('model_visualization: {}.png\n'.format(test_name))
    lines_list.append('epochs: {}\n'.format(epochs))
    lines_list.append('batch: {}\n'.format(batch))
    lines_list.append('loss: {}\n'.format(evaluation[0]))
    lines_list.append('mse: {}\n'.format(evaluation[1]))

    with open(os.path.join(path, 'all_logs_oneax.txt'.format(test_name)), 'a+') as f:
        f.writelines(lines_list)


if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense_2/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    time_stamp = datetime.datetime.now()
    time_string = '{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(time_stamp.year%100, time_stamp.month, time_stamp.day, time_stamp.hour, time_stamp.minute)
    print(time_string)

    epochs = 100
    batch = 100
    test_name = '4layer_square' + time_string
    data = data_prep.main(data_file)

    model = define_model()
    model, evaluation, history = training(model, data, epochs, batch)
    log()