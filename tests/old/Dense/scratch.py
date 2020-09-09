import os
import numpy as np


def get_best(_file, axis, n_top):
    read = np.load(infile)
    sorted_read = read[read[:, axis].argsort()[::-1]]
    return sorted_read[:n_top]

path = '/tests/old/Dense'
files = ['results1.npy', 'results2.npy', 'results3.npy']

for f in files:
    print(f)
    infile = os.path.join(path, f)
    n = get_best(infile, 3, 3)
    s = get_best(infile, 2, 3)
    print(n)
    print(s)
# write all

# print('skip-connection')
# print(read[read[:, 2].argsort()[::-1]])
# print('bez skip-connection')
# print(read[read[:, 3].argsort()[::-1]])


