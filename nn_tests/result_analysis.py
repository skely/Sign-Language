import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
sys.path.append('/home/jedle/Projects/Sign-Language/lib/')
import data_prep

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


def plot_compare(data, labels, title='', last=False, version=0):
    if version == 0:
        channels = ['X', 'Y', 'Z']
        plt.figure()
        if title is not '':
            plt.title(title)
        for d, l in zip(data, labels):
            for ch in range(3):
                plt.plot(d[:, ch], label=l+ ' ' +channels[ch])
        plt.legend()
        if last:
            plt.show()
    elif version == 1:  # neni dodelany
        channels = ['X', 'Y', 'Z']
        for ch in range(3):
            plt.figure()
            if title is not '':
                plt.title(title)
            for d, l in zip(data, labels):
                plt.plot(d[:, ch], label=l + ' ' + channels[ch])
            plt.legend()
        if last:
            plt.show()
    else:
        print('Specify version!')


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


def get_best_model(_path):
    test_file_list = get_all_testfiles(_path)
    all_results = read_all_test_files(test_file_list, _verbose=1)

    selected_results = [l for l in all_results if l['epochs'] == '3000'][0]
    return selected_results['model file name']


def plot_example(all_results, selection=0):
    selected_model_name = all_results[0]['model file name']
    model = load_model(os.path.join(path, selected_model_name))
    limit_prediction = selection+1

    data = data_prep.load_HDF5(os.path.join(path, data_file))
    train_X, train_Y, test_X, test_Y = data
    if 'simple' in selected_model_name:  # reshape 3D vystupu
        predicted = model.predict(test_X[:limit_prediction, :, 0:1])
    else:
        test_Y = np.reshape(test_Y, (-1, 97, 3), 'F')
        predicted = model.predict(test_X[:limit_prediction])
        predicted = np.reshape(predicted, (-1, 97, 3), 'F')

    plot_compare([test_X[selection, :], test_Y[selection, :], predicted[selection, :]], ['polynomial interpolation', 'ground truth', 'nn prediction'], title='prediction data', version=1)
    plot_compare([test_X[selection, :], test_Y[selection, :], predicted[selection, :]], ['polynomial interpolation', 'ground truth', 'nn prediction'], title='prediction data', version=0)


def plot_all_histories(_selected_results, left_cut=1):
    plt.figure()
    for tmp_item in _selected_results:
        # print(tmp_item)
        tmp_history_file = os.path.join(path, tmp_item['training history'])
        tmp_history = np.load(tmp_history_file, allow_pickle=True)
        plt.plot(tmp_history['loss'][left_cut:], label=tmp_item['test name'].split('_')[0] + ' ' + tmp_item['learning_rate'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # path = '/home/jedle/Projects/Sign-Language/tests/Conv1D/tests'
    path = '/home/jedle/Projects/Sign-Language/nn_tests/data'
    # path = '/home/jedle/Projects/Sign-Language/nn_tests/data_old'
    # data_file = '/home/jedle/data/Sign-Language/_source_clean/prepared_data_30-30_aug10times2.npz'
    # data_h5_file = '/home/jedle/Projects/Sign-Language/tests/old/Conv3D/tests/simple_aug10.h5'
    data_file = os.path.join(path, '3D_aug10.h5')
    # data_file = os.path.join('/home/jedle/Projects/Sign-Language/nn_tests/data', '3D_aug10.h5')
    epsilon = 10e-8
    selection = 50

    data = data_prep.load_HDF5(os.path.join(path, data_file))
    train_X, train_Y, test_X, test_Y = data
    test_Y = np.reshape(test_Y, (-1, 97, 3), 'F')

    # *** load logs
    test_file_list = get_all_testfiles(path)
    all_results = read_all_test_files(test_file_list, _verbose=1)

    # for i in all_results:
    #     print(i['test name'])
    # *** grand filter logs
    selected_results = all_results
    # print(selected_results[0].keys())

    selected_results = [l for l in selected_results if 'ihTi4' not in l['test name']]
    selected_results = [l for l in selected_results if 'eusRT' not in l['test name']]

    # *** plot custom loss histories
    best_representatives = selected_results
    # best_representatives = [l for l in best_representatives if 'train' not in l['test name']]
    # best_representatives = [l for l in best_representatives if 'oldcopypasta' not in l['test name']]

    plt.figure()
    plt.axhline(y=epsilon, linewidth=.5, color='r')
    plt.title('loss comparison (baseline={})'.format(epsilon))

    for tmp_item in best_representatives:
        tmp_history_file = os.path.join(path, tmp_item['training history'])
        tmp_history = np.load(tmp_history_file, allow_pickle=True)
        print('{} : {} : {} : {}'.format(tmp_item['test name'], tmp_item['loaded model file name (continuous training)'], tmp_item['learning_rate'], tmp_item['loss']))
        # print(tmp_history.keys())
        tmp_epochs = int(tmp_item['epochs'])
        label_string = tmp_item['test name'].split('-')[3]+ ' ' +tmp_item['test name'].split('_')[1] + ' ' + tmp_item['learning_rate']
        tmp_history_show = tmp_history['val_loss']
        if 'gen0' in tmp_item['test name']:
            plt.plot(np.arange(tmp_epochs)[1000:], tmp_history_show[1000:], label=label_string)
        else:
            gen_number = int(tmp_item['test name'].split('_')[1][3])
            # print(gen_number)
            gen_shift = gen_number * 3000
            # if 'gen1' in tmp_item['test name']:
            #     gen_shift = 3000
            # elif 'gen2' in tmp_item['test name']:
            #     gen_shift = 6000
            # elif 'gen3' in tmp_item['test name']:
            #     gen_shift = 9000
            # elif 'gen4' in tmp_item['test name']:
            #     gen_shift = 12000
            # elif 'gen5' in tmp_item['test name']:
            #     gen_shift = 15000
            plt.plot(np.arange(tmp_epochs) + gen_shift, tmp_history_show, label=label_string)
    plt.legend()

    # *** plot example
    # plot_example(best_representatives, selection=selection)
    plt.show()
