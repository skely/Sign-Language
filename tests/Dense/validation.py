import os
import numpy as np
import tests.Dense.model_predict as model_predict
import random
import matplotlib.pyplot as plt
from keras.models import load_model



def compare(_orig, _comp):
    mse = ((_orig - _comp) ** 2).mean(axis=1)
    # print(mse)
    # print(sum(mse))
    return np.mean(mse)


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
    return results


def evaluate_results(_results, n_best=1):
    sorted_dict = sorted(_results, key=lambda i: i['mse'], reverse=True)
    return sorted_dict[0:n_best]


def get_all_testfiles(_path):
    file_list = os.listdir(_path)
    test_file_list = [os.path.join(_path, f) for f in file_list if 'results_' in f]
    return test_file_list


def show_loss_history():
    pass


if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    get_all_testfiles(path)
    # test_file = os.path.join(path, 'results_dense_3layer_sgd_lr001_skips_bias_sigmoid.txt')


    # comparison of cubic interpolation and origina data
    # **************************************************
    data = np.load(data_file)
    orig = data['test_Y'][0]
    comp = data['test_X'][0]
    print('Comparison original and interpolated (cubic) data: {}'.format(compare(orig, comp)))

    # get the best model
    best_result = np.inf
    for test_file in get_all_testfiles(path):
        # print(test_file)
        test_results = read_test_file(test_file)
        test_results = evaluate_results(test_results)
        if float(test_results[0]['mse']) < best_result:
            best_result = float(test_results[0]['mse'])
            best_model = os.path.join(path, test_results[0]['model file name'])
        # print(test_results[0]['mse'])
        # print(test_results[0]['model file name'])

    # model_file = os.path.join(path, 'model_dense_3layer_run_3.h5')
    print(best_result)
    print(best_model)
    best_model = '/home/jedle/Projects/Sign-Language/tests/Dense/tests/model_dense_3layer_sgd_lr01_skips_bias_run_9.h5'
    print(path)
    print(data_file)
    print(best_model)
    model_predict.run_model_predict(path, data_file, best_model, 20, 1, train_data=False)

