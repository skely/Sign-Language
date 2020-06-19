import os
import pickle
import numpy as np
# import tests.Dense_2.data_prep as data_prep
import data_prep
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import sgd


def log_read(file):
    with open(file, 'r') as f:
        content = f.readlines()
    return content


def read_test_file(_test_file):
    with open(_test_file, 'r') as f:
        cont = f.readlines()
    results = []

    for line in cont:
        if '***' in line:
            if 'tmp_result' in locals():
                results.append(tmp_result)
            tmp_result = {}
        else:
            tmp_line = line.strip().split(': ')
            tmp_result[tmp_line[0]] = tmp_line[1]
        results.append(tmp_result)
    return results


def get_all_testfiles(_path):
    file_list = os.listdir(_path)
    test_file_list = [os.path.join(_path, f) for f in file_list if 'all_logs' in f]
    return test_file_list


if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense_2/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    test_file_list = get_all_testfiles(path)
    print(test_file_list)

    results = read_test_file(test_file_list[0])
    print(results)