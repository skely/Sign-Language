import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import adam, sgd

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


# def my_resize_dense_flat(_data):
#     batch, time, features = np.shape(_data)
#     _new_data = np.zeros((batch * features, time))
#     for b in range(batch):
#         for f in range(features):
#             _new_data[b * f:(b + 1) * f, :] = _data[b, :, f]
#     return _new_data
#
#
# def model_Dense_flat(_trainX, _trainY, _testX, _testY):
#     _model = Sequential()
#     _model.add(Dense(97, input_dim=97, activation='relu'))
#     _model.add(Dense(1000, activation='relu'))
#     _model.add(Dense(2000, activation='relu'))
#     _model.add(Dense(1000, activation='relu'))
#     _model.add(Dense(97, activation='relu'))
#     _model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#     _model.fit(_trainX, _trainY, epochs=500, batch_size=100)
#     _, accuracy = model.evaluate(_testX, _testY)
#     print(' accuracy: {:.2f} %'.format(accuracy * 100))
#     return _model
#
#
# def model_Dense_3d(_trainX, _trainY, _testX, _testY):
#     _model = Sequential()
#     _model.add(Dense(3, input_shape=(97, 3), activation='relu'))
#     _model.add(Dense(100, activation='relu'))
#     _model.add(Dense(200, activation='relu'))
#     _model.add(Dense(400, activation='relu'))
#     _model.add(Dense(200, activation='relu'))
#     _model.add(Dense(100, activation='relu'))
#     _model.add(Dense(3, activation='relu'))
#     _model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#     _model.summary()
#     _model.fit(_trainX, _trainY, epochs=100, batch_size=100)
#     _, accuracy = _model.evaluate(_testX, _testY)
#     print(' accuracy: {:.2f} %'.format(accuracy * 100))
#     return _model
#
#
# def model_Dense_3d_noseq(_trainX, _trainY, _testX, _testY):
#     input_layer = Input(shape=(97, 3))
#     dense_1 = Dense(3, activation='relu')(input_layer)
#     skip_0 = concatenate([input_layer, dense_1])
#     dense_2 = Dense(10, activation='relu')(skip_0)
#     skip_1 = concatenate([skip_0, dense_2])
#     dense_3 = Dense(20, activation='relu')(skip_1)
#     skip_2 = concatenate([skip_1, dense_3])
#     dense_4 = Dense(40, activation='relu')(skip_2)
#     skip_3 = concatenate([skip_2, dense_4])
#     dense_5 = Dense(20, activation='relu')(skip_3)
#     skip_4 = concatenate([skip_3, dense_5])
#     dense_6 = Dense(10, activation='relu')(skip_4)
#     skip_5 = concatenate([skip_4, dense_6])
#     dense_7 = Dense(3, activation='relu')(skip_5)
#
#     _model = Model(inputs=input_layer, outputs=dense_7, name='3d_dense_model')
#     _model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#     _model.summary()
#     _model.fit(_trainX, _trainY, epochs=100, batch_size=100)
#     _, accuracy = _model.evaluate(_testX, _testY)
#     print(' accuracy: {:.2f} %'.format(accuracy * 100))
#     return _model


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


def Dense_v1_3layer_noskip(_data, _loss, _optimizer, _learning_rate, _epochs, _batch_size, _hidden_layer_sizes):
    batch, time, features = np.shape(_data[0])
    if _optimizer == 'adam':
        _optimizer = adam(learning_rate=_learning_rate)

    input_layer = Input(shape=(time, features))
    dense_1 = Dense(features, activation='relu')(input_layer)
    dense_2 = Dense(_hidden_layer_sizes[0], activation='relu')(dense_1)
    dense_3 = Dense(features, activation='relu')(dense_2)

    _model = Model(inputs=input_layer, outputs=dense_3, name='Dense_v1_3layer_noskip')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    _model.summary()
    _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _, _accuracy = _model.evaluate(_data[2], _data[3])
    # print(' accuracy: {:.2f} %'.format(accuracy * 100))
    return _model, _accuracy


def Dense_v1_3layer(_data, _loss, _optimizer, _learning_rate, _epochs, _batch_size, _hidden_layer_sizes):
    batch, time, features = np.shape(_data[0])

    input_layer = Input(shape=(time, features))
    dense_1 = Dense(features, activation='relu')(input_layer)
    skip_1 = concatenate([input_layer, dense_1])
    dense_2 = Dense(_hidden_layer_sizes[0], activation='relu')(skip_1)
    skip_2 = concatenate([skip_1, dense_2])
    dense_3 = Dense(features, activation='relu')(skip_2)

    _model = Model(inputs=input_layer, outputs=dense_3, name='Dense_v1_3layer_noskip')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    _model.summary()
    _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _, _accuracy = _model.evaluate(_data[2], _data[3])
    print(' accuracy: {:.2f} %'.format(_accuracy * 100))
    return _model, _accuracy


def Dense_v1_4layer_noskip(_data, _loss, _optimizer, _learning_rate, _epochs, _batch_size, _hidden_layer_sizes):
    batch, time, features = np.shape(_data[0])

    input_layer = Input(shape=(time, features))
    dense_1 = Dense(features, activation='relu')(input_layer)
    dense_2 = Dense(_hidden_layer_sizes[0], activation='relu')(dense_1)
    dense_3 = Dense(_hidden_layer_sizes[1], activation='relu')(dense_2)
    dense_4 = Dense(features, activation='relu')(dense_3)

    _model = Model(inputs=input_layer, outputs=dense_4, name='Dense_v1_4layer_noskip')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    _model.summary()
    _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _, _accuracy = _model.evaluate(_data[2], _data[3])
    print(' accuracy: {:.2f} %'.format(_accuracy * 100))
    return _model, _accuracy


def Dense_v1_4layer(_data, _loss, _optimizer, _learning_rate, _epochs, _batch_size, _hidden_layer_sizes):
    batch, time, features = np.shape(_data[0])

    input_layer = Input(shape=(time, features))
    dense_1 = Dense(features, activation='relu')(input_layer)
    skip_1 = concatenate([input_layer, dense_1])
    dense_2 = Dense(_hidden_layer_sizes[0], activation='relu')(skip_1)
    skip_2 = concatenate([skip_1, dense_2])
    dense_3 = Dense(_hidden_layer_sizes[1], activation='relu')(skip_2)
    skip_3 = concatenate([skip_2, dense_3])
    dense_4 = Dense(features, activation='relu')(skip_3)

    _model = Model(inputs=input_layer, outputs=dense_4, name='Dense_v1_4layer_noskip')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    _model.summary()
    _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _, _accuracy = _model.evaluate(_data[2], _data[3])
    print(' accuracy: {:.2f} %'.format(_accuracy * 100))
    return _model, _accuracy


def Dense_v1_5layer_noskip(_data, _loss, _optimizer, _learning_rate, _epochs, _batch_size, _hidden_layer_sizes):
    batch, time, features = np.shape(_data[0])
    input_layer = Input(shape=(time, features))
    dense_1 = Dense(features, activation='relu')(input_layer)
    dense_2 = Dense(_hidden_layer_sizes[0], activation='relu')(dense_1)
    dense_3 = Dense(_hidden_layer_sizes[1], activation='relu')(dense_2)
    dense_4 = Dense(_hidden_layer_sizes[2], activation='relu')(dense_3)
    dense_5 = Dense(features, activation='relu')(dense_4)

    _model = Model(inputs=input_layer, outputs=dense_5, name='Dense_v1_5layer_noskip')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    _model.summary()
    _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _, _accuracy = _model.evaluate(_data[2], _data[3])
    print(' accuracy: {:.2f} %'.format(_accuracy * 100))
    return _model, _accuracy


def Dense_v1_5layer(_data, _loss, _optimizer, _learning_rate, _epochs, _batch_size, _hidden_layer_sizes):
    batch, time, features = np.shape(_data[0])

    input_layer = Input(shape=(time, features))
    dense_1 = Dense(features, activation='relu')(input_layer)
    skip_1 = concatenate([input_layer, dense_1])
    dense_2 = Dense(_hidden_layer_sizes[0], activation='relu')(skip_1)
    skip_2 = concatenate([skip_1, dense_2])
    dense_3 = Dense(_hidden_layer_sizes[1], activation='relu')(skip_2)
    skip_3 = concatenate([skip_2, dense_3])
    dense_4 = Dense(_hidden_layer_sizes[2], activation='relu')(skip_3)
    skip_4 = concatenate([skip_3, dense_4])
    dense_5 = Dense(features, activation='relu')(skip_4)

    _model = Model(inputs=input_layer, outputs=dense_5, name='Dense_v1_5layer_noskip')
    _model.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    _model.summary()
    _model.fit(_data[0], _data[1], epochs=_epochs, batch_size=_batch_size, verbose=1)
    _, _accuracy = _model.evaluate(_data[2], _data[3])
    print(' accuracy: {:.2f} %'.format(_accuracy * 100))
    return _model, _accuracy

if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    data = prepare_data_file(data_file)

    loss = 'mean_squared_error'
    learning_rate = 0.001
    optimizer = 'adam'

    epochs = 2
    batch_size = 100
    # hidden_layer_sizes = [[9], [27], [81], [248]]
    hidden_layer_sizes = [[9]]
    repetitions = 1
    results_filename = 'results3'
    results = []
    results.append('data_file={}'.format(data_file))
    results.append('loss={}'.format(loss))
    results.append('optimizer={}'.format(optimizer))
    results.append('learning_rate={}'.format(learning_rate))
    results.append('epochs={}'.format(epochs))
    results.append('batch_size={}'.format(batch_size))
    results.append('hidden_layer_sizes={}'.format(hidden_layer_sizes))
    results.append('repetitions={}'.format(repetitions))
    results.append('\n')

    with open(os.path.join(path, results_filename+'.txt') as f:
	f.wirtelines(results)	
    results = []

    for h in hidden_layer_sizes:
        for r in range(repetitions):
            model_name = 'model_Dense_v1_{}_r{}'.format(h[0], r)
            model_s, accuracy_s = Dense_v1_3layer_noskip(data, loss, optimizer, learning_rate, epochs, batch_size, h)
            model_n, accuracy_n = Dense_v1_3layer(data, loss, optimizer, learning_rate, epochs, batch_size, h)
            results.append([h, r, accuracy_s, accuracy_n])
            model_s.save(os.path.join(path, model_name+'s'))
            model_n.save(os.path.join(path, model_name+'n'))

    print(results)
    np.save(os.path.join(path, results_filename), results)







