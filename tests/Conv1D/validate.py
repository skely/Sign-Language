import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import tests.Dense_2.data_prep as data_prep
import data_prep
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import sgd


def log_read(file):
    with open(file, 'r') as f:
        content = f.readlines()
    return content

def read_all_test_files(_test_file_list, _verbose=0):
    _results = []
    for tst_file in _test_file_list:
        _new_res = read_test_file(tst_file)
        _results += _new_res
    return _results


def read_test_file(_test_file, _verbose=0):
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
    if _verbose == 1:
        for tmp_res in results:
            print('*****')
            print('test name: {}'.format(tmp_res['test name']))
            print('model file name: {}'.format(tmp_res['model file name']))
            print('mse: {}'.format(tmp_res['mse']))
    else:
        pass
    return results


def evaluate_results(_results, n_best=-1):
    sorted_dict = sorted(_results, key=lambda i: float(i['mse']), reverse=False)
    if n_best == -1:
        return sorted_dict
    else:
        return sorted_dict[0:n_best]


def get_all_testfiles(_path):
    file_list = os.listdir(_path)
    test_file_list = [os.path.join(_path, f) for f in file_list if 'all_logs' in f]
    return test_file_list


def plot_compare(data, labels):
    plt.figure()
    for d, l in zip(data, labels):
        plt.plot(d, label=l)
    plt.legend()
    # plt.show()


def training_loss_graph(_result):
    # print(_result['training history'])
    history = np.load(os.path.join(path, _result['training history']), allow_pickle=True)
    # history = np.asarray(history)

    loss_history = history['loss']
    epsilon_threshold = 0
    if 'epsilon' in globals():
        for i in range(len(loss_history)):
            if i > 0:
                if abs(loss_history[i] - loss_history[i-1]) > epsilon:
                    # print(abs(loss_history[i] - loss_history[i-1]))
                    epsilon_threshold = i
    plt.figure()
    plt.title('loss history ({})'.format(epsilon))
    plt.plot(loss_history, label='training loss')
    plt.axvline(x=epsilon_threshold, color='r', label='epsilon_thr={}'.format(epsilon_threshold))
    plt.legend()




if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Conv1D/tests'
    data_file = '/home/jedle/data/Sign-Language/_source_clean/testing/prepared_data_glo_30-30.npz'

    epsilon = 10e-8

    test_file_list = get_all_testfiles(path)
    results = read_all_test_files(test_file_list, _verbose=1)
    data = data_prep.main(data_file)

    ordered = evaluate_results(results)
    for o in ordered:
        print('{} - mse: {}'.format(o['model file name'], o['mse']))

    picked_result = ordered[0]

    training_loss_graph(picked_result)

    model = load_model(os.path.join(path, picked_result['model file name']))
    # model.summary()
    predicted = model.predict(data[0][0:10, :])

    # plot_compare([data[0][0, :], data[1][0, :]], ['interpolated', 'ground truth'])
    # plot_compare([data[0][0, :], predicted[0, :]], ['interpolated', 'predicted'])
    plot_compare([data[0][0, :], data[1][0, :], predicted[0, :]], ['interpolated', 'ground truth', 'predicted'])
    plt.show()