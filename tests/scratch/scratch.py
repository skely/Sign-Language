from lib import BVH
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    arr = np.ones((1406, 97, 183))
    print(np.shape(arr))
    expansion = np.zeros((1406, 1, 183))

    print(np.shape(expansion))

    print(np.shape(np.concatenate((arr, expansion), 1)))
