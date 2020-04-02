import os
import json
import numpy as np
import fastdtw
import matplotlib.pyplot as plt
from lib import data_prep, BVH


if __name__ == '__main__':
    working_folder = '/home/jedle/data/Sign-Language/BVH_synthesis_rev2/temp'
    file_list = os.listdir(working_folder)
    file_templates = ['_'.join(f.split('_')[1:]) for f in file_list if 'synth' in f]
    for file_name in file_templates:
        template_file = '/home/jedle/data/Sign-Language/BVH_synthesis_rev2/temp/orig_{}'.format(file_name)
        comparison_file = '/home/jedle/data/Sign-Language/BVH_synthesis_rev2/temp/synth_{}'.format(file_name)

        template_trajectory = BVH.load_trajectory(template_file)
        comparison_trajectory = BVH.load_trajectory(comparison_file)

        comparison, path = data_prep.sign_comparison(template_trajectory[:, :], comparison_trajectory[:, :])
        print('src_file: {}'.format(file_name))
        data_shape = (np.shape(template_trajectory))
        print('length: {}'.format(data_shape[0]))
        print('distance: {:.2f}'.format(comparison/data_shape[1]))
        print('---')

        print(np.shape(comparison_trajectory))

        plt.plot(template_trajectory[:, 24:27], 'b')
        plt.plot(comparison_trajectory[:, 24:27], 'r')
        plt.show()
