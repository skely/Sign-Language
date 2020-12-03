import os
import numpy as np
from lib import data_prep, BVH
import matplotlib.pyplot as plt

source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
template_bvh_file = os.path.join(source_dir, 'bvh/16_05_20_a_R.bvh')
new_data_file = os.path.join(source_dir, 'prepared_data_30-30_aug10times2.npz')

loaded_data = np.load(new_data_file)
data_focus = loaded_data['train_X'][0]

BVH.generate_BVH(data_focus, 'nonzero_base.bvh', template_bvh_file, channels='rotation', zero_shift=False)