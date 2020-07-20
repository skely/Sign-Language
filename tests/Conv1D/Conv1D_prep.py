import numpy as np
import matplotlib.pyplot as plt

data_file = '/home/jedle/data/Sign-Language/_source_clean/prepared_data_ang_30-30_aug10times.npz'
data = np.load(data_file)

train_X = data['train_X']
for i in range(10):
    plt.plot(train_X[i, :, 0])
    # plt.plot(train_X[i, :, 1])
    # plt.plot(train_X[i, :, 2])
    plt.show()