import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
# import tests.Dense_2.data_prep as data_prep
from lib import data_prep as dp
import data_prep
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import sgd


def log_read(file):
    with open(file, 'r') as f:
        content = f.readlines()
    return content

def read_all_test_files(_test_file_list, _verbose=0, _evaluate=True):
    _results = []
    for tst_file in _test_file_list:
        _new_res = read_test_file(tst_file)
        _results += _new_res
    if _evaluate:
        _results = evaluate_results(_results)
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


def plot_compare(data, labels, title=''):
    plt.figure()
    if title is not '':
        plt.title(title)
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
    plt.title('loss history')
    plt.plot(loss_history, label='training loss')
    plt.axvline(x=epsilon_threshold, color='r', label='epsilon_thr=({}) - {}'.format(epsilon, epsilon_threshold))
    plt.legend()




if __name__ == '__main__':
    path = '/home/jedle/Projects/Sign-Language/tests/Conv1D/tests'
    # data_file = '/home/jedle/data/Sign-Language/_source_clean/prepared_data_30-30_aug10times2.npz'
    data_h5_file = '/home/jedle/Projects/Sign-Language/tests/Conv1D/tests/prepared_data_ang_aug10.h5'

    epsilon = 10e-8

    # select model
    test_file_list = get_all_testfiles(path)
    for t in test_file_list:
        print(t)
    results = read_all_test_files(test_file_list, _verbose=1)
    for line in results:
        print(line)
    selected_model_name = results[0]['model file name']
    selected_model_history = results[0]['training history']
    model = load_model(os.path.join(path, selected_model_name))
    history = np.load(os.path.join(path, selected_model_history), allow_pickle=True)
    print(history.keys())
    plt.figure()
    plt.title('Loss progress')
    plt.plot(history['loss'][10:])
    plt.show()

    loss_diff = np.diff(history['loss'])
    for i in range(len(loss_diff)):
        print(loss_diff[i])

    doit = False
    if doit:
        # select data
        data = data_prep.load_HDF5(os.path.join(path, 'prepared_data_ang_aug10.h5'))
        train_X, train_Y, test_X, test_Y = data
        predicted = model.predict(test_X[0:10])

        # evaluate
        test_Y = data_prep.conv_1d23d(test_Y)
        for i in range(np.size(test_X), 0):
            pass
        dist_ground = dp.sign_comparison(test_X[0, :, 0:1], test_Y[0, :, 0:1])
        plot_compare([test_X[0, :, 0], test_Y[0, :, 0]], ['test_X', 'test_Y'], title='source data')

        predicted = data_prep.conv_1d23d(predicted)
        dist_predicted = dp.sign_comparison(test_X[0, :, 0:1], predicted[0, :, 0:1])
        plot_compare([test_X[0, :, 0], predicted[0, :, 0]], ['test_X', 'prediction'], title='prediction data')

        print('error ground: {:.8f}'.format(dist_ground))
        print('error predicted: {:.8f}'.format(dist_predicted))
        print('error ratio: {:.4f}'.format(dist_predicted/dist_ground))
        plt.show()

