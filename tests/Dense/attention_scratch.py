import os, sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense

data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'
orig_data = np.load(data_file)

train_X = orig_data['train_X']#[:, :, 21]
train_Y = orig_data['train_Y']#[:, :, 21]

train = False
test = False
if train:

    model = Sequential()
    model.add(Dense(183, input_shape=(97, 183), activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(3000, activation='relu'))
    model.add(Dense(5000, activation='relu'))
    model.add(Dense(3000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(183, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(train_X, train_Y, epochs=10, batch_size=10)
    _, accuracy = model.evaluate(orig_data['test_X'], orig_data['test_Y'])
    print('accuracy: {:.2f}%'.format(accuracy*100))

    model.save('model_v0')
if test:
    loaded = load_model('model_v0')
    response = loaded.predict(orig_data['test_X'])
    plt.plot(response[0, :, 21], label='predict')
    plt.plot(orig_data['test_Y'][0, :, 21], label='orig')
    plt.legend()
    plt.show()

