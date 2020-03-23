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
    for l in log:
        # print(l)
        kernels.append(l['kernels'])
        epochs.append(l['epochs'])
        loss.append(l['loss'])
        val_loss.append(l['val_loss'])

    order = np.argsort(loss)

    print('{:<10} {:<10} {:<10} {:<10}'.format('kernels', 'epochs', 'loss', 'val_loss'))
    for i in order:
        print('{:<10} {:<10} {:<10.5f} {:<10.5f}'.format(kernels[i], epochs[i], loss[i], val_loss[i]))


if __name__ == '__main__':
    logfile = '/home/jedle/data/Sign-Language/_source_clean/testing/test_v1/losses.txt'

    log = read_logfile(logfile)
    sort_results_by_loss(log)
