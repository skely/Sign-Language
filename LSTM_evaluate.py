import os
import numpy as np
import matplotlib.pyplot as plt
import json


def read_logfile(_log_file):
    with open(_log_file, 'r') as jf:
        _log = json.load(jf)

    return _log


def sort_results_by_loss(_log):
    kernels = []
    epochs = []
    loss = []
    val_loss = []
    for l in _log:
        # print(l)
        kernels.append(l['kernels'])
        epochs.append(l['epochs'])
        loss.append(l['loss'])
        val_loss.append(l['val_loss'])

    order = np.argsort(loss)

    print('{:<10} {:<10} {:<10} {:<10}'.format('kernels', 'epochs', 'loss', 'val_loss'))
    for i in order:
        print('{:<10} {:<10} {:<10.5f} {:<10.5f}'.format(kernels[i], epochs[i], loss[i], val_loss[i]))


def test_loss_comparison(_log_dir):
    dirlist = os.listdir(_log_dir)
    for item in dirlist:
        tmp = os.path.join(_log_dir, item)
        if os.path.isdir(tmp):
            print(tmp)
            logfile = os.path.join(tmp, 'losses.txt')
            log = read_logfile(logfile)
            sort_results_by_loss(log)


if __name__ == '__main__':
    tmp_dir = '/home/jedle/data/Sign-Language/_source_clean/testing/'
    test_loss_comparison(tmp_dir)
