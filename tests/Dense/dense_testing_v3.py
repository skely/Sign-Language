import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import adam, sgd
from keras.utils import plot_model


def load_data_file(_data_file):
    _data = np.load(_data_file)
    _train_X = _data['train_X']
    _train_Y = _data['train_Y']
    _test_X = _data['test_X']
    _test_Y = _data['test_Y']
    return _train_X, _train_Y, _test_X, _test_Y


def data_reshape(_data):
    batch, time, channels = np.shape(_data)
    reshaped_data = np.zeros((batch*channels, time))
    for i in range(batch):
        reshaped_data[i*channels:(i+1)*channels, :] = np.transpose(_data[i, :, :])
    return reshaped_data


def data_normalization_get_min_max(_data):
    _min = np.min(_data)
    _max = np.max(_data)
    return _min, _max


def data_normalization(_data):
    _min, _max = data_normalization_get_min_max(_data)
    _normed_data = (_data - _min) / (_max - _min)
    return _normed_data, [_min, _max]


def define_model():
    _loss = 'mean_squared_error'
    _optimizer = 'sgd'

    inputs = Input(shape=(97,))
    layer_1 = Dense(97, activation='sigmoid', use_bias=True)(inputs)
    layer_2 = Dense(97*3, activation='sigmoid', use_bias=True)(layer_1)
    layer_3 = Dense(97*9, activation='sigmoid', use_bias=True)(layer_2)
    layer_4 = Dense(97*3, activation='sigmoid', use_bias=True)(layer_3)
    layer_5 = Dense(97, activation='sigmoid', use_bias=True)(layer_4)

    _model = Model(inputs=inputs, outputs=layer_5, name='model_flat')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['mean_squared_error', 'accuracy'])
    _model.summary()

    return _model


def training(_model, _data, _epochs, _batch_size):
    _history = _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=2)
    _evaluations = _model.evaluate(_data[2], _data[3])
    return _model, _evaluations, _history


def make_log(log_file, test_name, evals):
    # model_plot_name = 'model_{}_run_{}.png'.format(test_name, i)
    # model_name = 'model_{}_run_{}.h5'.format(test_name, i)
    #
    # plot_model(model, to_file=os.path.join(path, model_plot_name), show_shapes=True)
    #
    lines_list = []
    lines_list.append('*******************************\n')

    _, mse, accuracy = evals

    lines_list.append('test name: {}\n'.format(test_name))
    # lines_list.append('model file name: {}\n'.format(model_name))
    #
    # lines_list.append('hidden_layers: {}\n'.format(h))
    # lines_list.append('skips: {}\n'.format(skips))
    # lines_list.append('model plot: {}\n'.format(test_name + '.png'))
    #
    # lines_list.append('loss function: {}\n'.format(loss))
    # lines_list.append('optimizer_name: {}\n'.format(optimizer_name))
    # lines_list.append('learning_rate: {}\n'.format(learning_rate))
    # lines_list.append('momentum: {}\n'.format(momentum))
    # lines_list.append('decay: {}\n'.format(decay))
    #
    # lines_list.append('epochs: {}\n'.format(epochs))
    # lines_list.append('batch_size: {}\n'.format(batch_size))
    #
    lines_list.append('mse: {}\n'.format(mse))
    lines_list.append('accuracy: {}\n'.format(accuracy))
    # lines_list.append('start_time: {}\n'.format(start_time))
    # lines_list.append('end_time: {}\n'.format(end_time))
    # lines_list.append('duration: {}\n'.format(end_time-start_time))
    #
    with open(log_file, 'a+') as f:
        f.writelines(lines_list)


def main():
    path = '/home/jedle/Projects/Sign-Language/tests/Dense/tests_flat'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    test_name = 'v0'

    data = load_data_file(data_file)
    train_X = data_reshape(data[0])
    train_Y = data_reshape(data[1])
    test_X = data_reshape(data[2])
    test_Y = data_reshape(data[3])
    data_for_training = [train_X, train_Y, test_X, test_Y]

    _data_concat = np.concatenate(data_for_training)
    _data_concat_normed, _ = data_normalization(_data_concat)
    sizes = []
    for i in range(len(data_for_training)):
        if i == 0 :
            sizes.append(np.shape(data_for_training[i])[0])
        else:
            sizes.append((np.shape(data_for_training[i])[0]) + sizes[len(sizes) - 1])

    print(sizes)
    train_X = _data_concat_normed[0:sizes[0], :]
    train_Y = _data_concat_normed[sizes[0]:sizes[1], :]
    test_X = _data_concat_normed[sizes[1]:sizes[2], :]
    test_Y = _data_concat_normed[sizes[2]:, :]
    data_for_training = [train_X, train_Y, test_X, test_Y]

    # plt.plot(data_for_training[0][0, :])
    # plt.plot(data_for_training[1][0, :])
    # plt.show()

    model = define_model()
    model, evals, history = training(model, data_for_training, 100, 100)

    make_log(os.path.join(data_file, 'results.txt'), test_name, evals)

if __name__ == '__main__':
    main()