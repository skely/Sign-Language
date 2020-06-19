import os
import pickle
import numpy as np
# import tests.Dense_2.data_prep as data_prep
import data_prep
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import sgd

def define_model():
    _loss = 'mean_squared_error'
    _optimizer = 'sgd'
    lr = 0.01
    momentum = 0.
    decay = 1e-8/epochs
    _optimizer = sgd(lr, momentum, decay=decay)
    _activation = 'sigmoid'

    inputs = Input(shape=(97,))
    layer_1 = Dense(97, activation=_activation, use_bias=True)(inputs)
    layer_2 = Dense(97*3, activation=_activation, use_bias=True)(layer_1)
    layer_3 = Dense(97*9, activation=_activation, use_bias=True)(layer_2)
    layer_4 = Dense(97*27, activation=_activation, use_bias=True)(layer_3)
    layer_5 = Dense(97*9, activation=_activation, use_bias=True)(layer_4)
    layer_6 = Dense(97*3, activation=_activation, use_bias=True)(layer_5)
    layer_7 = Dense(97, activation=_activation, use_bias=True)(layer_6)

    _model = Model(inputs=inputs, outputs=layer_7, name='model_flat')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
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

    with open(os.path.join(path, 'all_logs.txt'.format(test_name)), 'a+') as f:
        f.writelines(lines_list)


if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense_2/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    epochs = 200
    batch = 1000
    test_name = 'pyramid_7l_decay'
    data = data_prep.main(data_file)
    model = define_model()
    model, evaluation, history = training(model, data, epochs, batch)
    log()


