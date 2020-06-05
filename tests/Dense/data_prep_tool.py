import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense

def data_normalization(_data):
    _min = np.min(_data)
    _max = np.max(_data)

    _normed_data = (_data - _min) / (_max - _min)
    return _normed_data, [_min, _max]


def my_resize_dense_3d(_data):
    new_feat = 3
    batch, time, features = np.shape(_data)
    _new_data = np.zeros((batch * int(features/new_feat), time, new_feat))

    for b in range(batch):
        for f in range(int(features/new_feat)):
            _new_data[b * f:(b + 1) * f, :, :] = _data[b, :, f*3:(f+1)*3]
    return _new_data


def my_resize_dense_flat(_data):
    batch, time, features = np.shape(_data)
    _new_data = np.zeros((batch*features, time))
    for b in range(batch):
        for f in range(features):
            _new_data[b*f:(b+1)*f, :] = _data[b, :, f]
    return _new_data


def model_Dense_flat(_trainX, _trainY, _testX, _testY):
    _model = Sequential()
    _model.add(Dense(97, input_dim=97, activation='relu'))
    _model.add(Dense(1000, activation='relu'))
    _model.add(Dense(2000, activation='relu'))
    _model.add(Dense(1000, activation='relu'))
    _model.add(Dense(97, activation='relu'))
    _model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    _model.summary()
    _model.fit(_trainX, _trainY, epochs=500, batch_size=100)
    _, accuracy = _model.evaluate(_testX, _testY)
    print(' accuracy: {:.2f} %'.format(accuracy * 100))
    return _model


if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    # model_name = 'model_v1'
    model_name = 'model_v1'

    train = True
    test = True

    data = np.load(data_file)
    train_X, minmax = data_normalization(data['train_X'])
    train_Y, minmax = data_normalization(data['train_Y'])
    test_X, minmax = data_normalization(data['test_X'])
    test_Y, minmax = data_normalization(data['test_Y'])

    train_X = my_resize_dense_flat(train_X)
    train_Y = my_resize_dense_flat(train_Y)
    test_X = my_resize_dense_flat(test_X)
    test_Y = my_resize_dense_flat(test_Y)
    #
    # train_X = my_resize_dense_3d(train_X)
    # train_Y = my_resize_dense_3d(train_Y)
    # test_X = my_resize_dense_3d(test_X)
    # test_Y = my_resize_dense_3d(test_Y)

    if train:

        model = model_Dense_flat(train_X, train_Y, test_X, test_Y)
        model.save(os.path.join(path, model_name))

    if test:
        item = 100
        loaded = load_model(os.path.join(path, model_name))
        response = loaded.predict(test_X)
        plt.plot(response[item, :], label='predict')
        plt.plot(test_X[item, :], label='input')
        plt.plot(test_Y[item, :], label='orig')
        plt.legend()
        plt.show()




