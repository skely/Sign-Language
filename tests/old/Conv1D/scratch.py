import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
# import tests.Dense_2.data_prep as data_prep
import data_prep
from keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, concatenate
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import sgd


def define_model():
    _loss = 'mean_squared_error'
    _optimizer = 'sgd'
    _optimizer = sgd(lr, momentum, decay=decay)
    _activation = 'sigmoid'

    # _model = Sequential()
    # _model.add(Conv1D(filters=64, kernel_size=3, activation=_activation, input_shape=(97, 1)))
    # _model.add(Conv1D(filters=64, kernel_size=3, activation=_activation))
    # _model.add(Flatten())
    # _model.add(Dense(97, activation=_activation))

    input = Input(shape=(97, 3))
    layer1 = Conv1D(filters=64, kernel_size=3, activation=_activation, padding='same')(input)
    concat1 = concatenate([input, layer1])
    layer2 = Conv1D(filters=64, kernel_size=5, activation=_activation, padding='same')(concat1)
    concat2 = concatenate([concat1, layer2])
    layer3 = Conv1D(filters=64, kernel_size=7, activation=_activation, padding='same')(concat1)
    concat3 = concatenate([concat2, layer3])
    layer4 = Flatten()(concat3)
    layer5 = Dense(97*3, activation=_activation)(layer4)


    _model = Model(inputs=input, outputs=layer5, name=_model_name)
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error'])
    _model.summary()

    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], validation_data=(_data[2], _data[3]), epochs=_epochs, batch_size=_batch_size, verbose=2)
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
    lines_list.append('learning_rate: {}\n'.format(lr))
    lines_list.append('decay: {}\n'.format(decay))

    with open(os.path.join(path, 'all_logs_oneax.txt'.format(test_name)), 'a+') as f:
        f.writelines(lines_list)


if __name__ == '__main__':
    path = '/tests/old/Conv1D/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    time_stamp = datetime.datetime.now()
    time_string = '{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(time_stamp.year%100, time_stamp.month, time_stamp.day, time_stamp.hour, time_stamp.minute)
    # print(time_string)
    _model_name = 'Conv3D_skips'

    epochs = 1000
    batch = 1000

    lr = 1e-2
    momentum = 0.9
    decay = 1e-2 / epochs

    test_name = '3Dconv_skip_' + time_string
    data = data_prep.main_3d(data_file)
    print(np.shape(data[0]))
    print(np.shape(data[1]))

    resized = data_prep.conv_1d23d(data[1])

    plt.plot(data[0][20, :, 2])
    plt.plot(resized[20, :, 2])
    plt.show()
    # model = define_model()
    # model, evaluation, history = training(model, data, epochs, batch)
    # log()