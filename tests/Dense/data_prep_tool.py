import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense

def data_normalization(_data):
    _min = np.min(_data)
    _max = np.max(_data)

    _normed_data = (_data - _min) / (_max - _min)
    return _normed_data


def my_resize(_data):
    batch, time, features = np.shape(_data)
    _new_data = np.zeros((batch*features, time))
    for b in range(batch):
        for f in range(features):
            _new_data[b*f:(b+1)*f, :] = _data[b, :, f]
    return _new_data

if __name__ == '__main__':
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
    data = np.load(data_file)

    train_X = data_normalization(data['train_X'])
    train_Y = data_normalization(data['train_Y'])
    test_X = data_normalization(data['test_X'])
    test_Y = data_normalization(data['test_Y'])

    train_X = my_resize(train_X)
    train_Y = my_resize(train_Y)
    test_X = my_resize(test_X)
    test_Y = my_resize(test_Y)

    plt.plot(train_X[0, :, 21])
    plt.plot(train_Y[0, :, 21])
    plt.show()

    # for b in range(np.size(train_X, 0)):
    #     plt.plot(train_X2[b, :])
    #     plt.plot(train_X[0, :, b])
    #     plt.show()

    # train_X = my_resize(train_X)
    # plt.plot(train_X[0:5, :])

    train = False
    test = True
    if train:
        model = Sequential()
        model.add(Dense(1000, input_dim=97, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(97, activation='relu'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        model.fit(train_X, train_Y, epochs=100, batch_size=100)
        _, accuracy = model.evaluate(test_X, test_Y)
        print('accuracy: {:.2f}%'.format(accuracy * 100))

        model.save('model_v0')
    if test:
        loaded = load_model('model_v0')
        response = loaded.predict(test_X)
        plt.plot(response[21, :], label='predict')
        plt.plot(test_Y[21, :], label='orig')
        plt.legend()
        plt.show()




