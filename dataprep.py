from lib import BVH, data_prep
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py


# data_dir = '/home/jedle/data/Sign-Language/_source_clean/dataprep/'
source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
bvh_dir = '/home/jedle/data/Sign-Language/_source_clean/bvh/'
# template for BVH structure
bvh_src_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_20_a_R.bvh'

dict_file = os.path.join(source_dir, 'ultimate_dictionary2.txt')
# take_dict_file = os.path.join(source_dir, 'dictionary_takes_v3.txt')
# dict_dict_file = os.path.join(source_dir, 'dictionary_dict_v4.txt')
# prepared_data_file = os.path.join(source_dir, 'prepared_data_ang_30-30_aug10_b.npz')
prepared_data_file = os.path.join(source_dir, 'prepared_data_ang_30-30_aug20.h5')
# augmented_data_file = os.path.join(source_dir, 'pure_30-30_aug10.h5')

m, c, _ = BVH.get_joint_list(bvh_src_file)
selected_sign_id = ''  # transitions
selected_sign_name = 'tra.'
# selected_file = '16_05_20_a_R.bvh'
selected_channels = 'rotation'  # kanály obsahující pouze rotace
surroundings = [30, 30]
transition_length = 37
random_seed = 9
train_test_ratio = 0.2

# SIGN SELECTION
prep = True
if prep:
    ch_sel_idxs = BVH.get_joint_id(m, c, '', _channel_name=selected_channels)
    ret, _ = data_prep.mine_sign_trajectories(bvh_dir, dict_file, _surroundings=surroundings, _sign_id=selected_sign_id, _sign_name=selected_sign_name, _channels=ch_sel_idxs, _verbose=True)

    item_list = []
    for item in ret:
        item_surrless = item[surroundings[0]: -surroundings[1], :]
        resized_kernel = data_prep.resample_trajectory(item_surrless, int(transition_length))
        new_item = np.concatenate((item[:surroundings[0]], resized_kernel, item[-surroundings[1]:]))
        item_list.append(new_item)
    item_list = np.asarray(item_list)

# ***** ANALYSE
# print(np.shape(item_list))
# batch, time, feature = np.shape(item_list)
# for i in range(batch):
#     for j in range(feature):
#         print(np.sum(item_list[i, :, j]))
#         print(np.average(item_list[i, :, j]))

# ***** add noisy items to list *****
noise = 0.01  # 0.1%
augmentation = 20

item_list = data_prep.augmentation_noise(item_list, augmentation, noise)

# ***** normalize *****
angular_limits = np.array((-360, 360))
# analysis
do_analysis = False
if do_analysis:
    print(np.shape(item_list))
    suspicious_channels = []
    for channel in range(np.size(item_list, 2)):
        minimum = np.min(item_list[:, :, channel])
        maximum = np.max(item_list[:, :, channel])
        if minimum < angular_limits[0] or maximum > angular_limits[1]:
            print(channel, minimum, maximum)
            suspicious_channels.append(channel)
    for susp in suspicious_channels:
        print(BVH.get_joint_name(m, c, susp))
        for i in range(np.size(item_list, 0)):
            minimum = np.min(item_list[i, :, susp])
            maximum = np.max(item_list[i, :, susp])
            if minimum < angular_limits[0] or maximum > angular_limits[1]:
                print(i, minimum, maximum, maximum - minimum)

# normalization (fuck the outliers)
# normalization to (-360, 360) -> (0-1)
_norm_scale = np.array([angular_limits[0], angular_limits[1]])
item_list = data_prep.normalize(item_list, _norm_scale)

# data - label split
# NN input data (data_X) : cubic approximation of trajectories

# *** augmented data save
# hf = h5py.File(augmented_data_file, 'w')
# hf.create_dataset('data', data=item_list)
# hf.close()

# *** train - test data preparation (interpolation of transition)
data_Y = item_list.copy()
data_X = item_list.copy()

tot_len = len(item_list)
print('Train - test split and preparations:')
for i, item in enumerate(item_list):
    item_bef = item[:surroundings[0]+1, :]
    item_aft = item[-surroundings[1]-1:, :]
    transition_length = np.size(item, 0) - surroundings[0] - surroundings[1]
    transition_approximation = data_prep.sign_synthesis(item_bef, item_aft, transition_length - 1)
    new_item = np.concatenate((item_bef, transition_approximation, item_aft))
    data_X[i, :, :] = new_item
    sys.stdout.write('\rDataprep processing... {:.2f}% done.'.format(100 * (i + 1) / tot_len))
sys.stdout.write('\rdone.\n')

# shuffle
data_Y = data_prep.shuffle(data_Y, random_seed)
data_X = data_prep.shuffle(data_X, random_seed)

# train - test split
split_pos = int(np.size(data_X, 0)*train_test_ratio)

train_Y = data_Y[split_pos:, :, :]
test_Y = data_Y[:split_pos, :, :]
train_X = data_X[split_pos:, :, :]
test_X = data_X[:split_pos, :, :]

# h5file = h5py.File(prepared_data_file, 'w')
# h5file.create_dataset('train_X', data=train_X)
# h5file.create_dataset('test_X', data=test_X)
# h5file.create_dataset('train_Y', data=train_Y)
# h5file.create_dataset('test_Y', data=test_Y)
np.savez(prepared_data_file, train_Y=train_Y, test_Y=test_Y, train_X=train_X, test_X=test_X)
