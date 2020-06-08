import os
import numpy as np
import pickle

def compare():
    data = np.load(data_file)

    orig = data['test_Y'][0]
    comp = data['test_X'][0]

    mse = ((orig - comp) ** 2).mean(axis=1)
    # print(mse)
    print(sum(mse))
    print(np.average(mse))


path = '/home/jedle/Projects/Sign-Language/tests/Dense/tests'
data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

hist_file = '/home/jedle/Projects/Sign-Language/tests/Dense/tests/history_dense_3layer_run_0.pkl'

with open(hist_file, 'rb') as f:
    ret = pickle.load(f)

print(ret)