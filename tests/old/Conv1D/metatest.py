import os
import pickle
import datetime
import numpy as np
from keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, concatenate
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import sgd


if __name__ == '__main__':
    print('hello world')
