import os
import numpy as np
import pickle

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

if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Dense/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    test_file = os.path.join(path, 'dense_3layer.txt')


    # comparison of cubic interpolation and origina data
    data = np.load(data_file)
    orig = data['test_Y'][0]
    comp = data['test_X'][0]

    print('Comparison original and interpolated (cubic) data: {}'.format(compare(orig, comp)))

    # get the best model
    test_results = read_test_file(test_file)
    print(test_results[0])
    model_file = os.path.join(path, 'model_dense_3layer_run_3.h5')

