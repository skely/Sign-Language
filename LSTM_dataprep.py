import BVH
import SL_dict
import data_prep
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/jedle/data/Sign-Language/_source_clean/dataprep/'
source_dir = '/home/jedle/data/Sign-Language/_source_clean/'
bvh_dir = '/home/jedle/data/Sign-Language/_source_clean/bvh/'
bvh_src_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_20_a_R.bvh'
take_dict_file = os.path.join(source_dir, 'dictionary_takes_v3.txt')
dict_dict_file = os.path.join(source_dir, 'dictionary_dict_v3.txt')

# dic = SL_dict.read_raw(take_dict_file)

m, c, _ = BVH.get_joint_list(bvh_src_file)
selected_sign_id = ''  # transitions
selected_sign_name = 'tra.'
selected_file = '16_05_20_a_R.bvh'
selected_channels = 'rotation'  # kanály obsahující pouze rotace
surroundings = [20, 20]
prep = False
if prep:
    ch_sel_idxs = BVH.get_joint_id(m, c, '', _channel_name='rotation')
    ret, _ = data_prep.mine_sign_trajectories(bvh_dir, take_dict_file, _surroundings=surroundings, _sign_id=selected_sign_id, _sign_name=selected_sign_name, _verbose=True)
    tot_len = 0
    for item in ret:
        # print(np.shape(item))
        tot_len += np.size(item, 0) - surroundings[0] - surroundings[1]
    avg_len = tot_len/len(ret)
    # print(avg_len)

    item_list = []
    for item in ret:
        item_surrless = item[surroundings[0]: -surroundings[1], :]
        resized_kernel = data_prep.resample_trajectory(item_surrless, int(avg_len))
        new_item = np.concatenate((item[:surroundings[0]], resized_kernel, item[-surroundings[1]:]))
        # print(np.shape(new_item))
        item_list.append(new_item)

    np.save(os.path.join(data_dir, 'transitions_sur{}'.format(surroundings[0])), item_list)

item_list = np.load(os.path.join(data_dir, 'transitions_sur{}.npy'.format(surroundings[0])))

# ***** normalize *****
# analysis
do_analysis = False
if do_analysis:
    print(np.shape(item_list))
    suspicious_channels = []
    for channel in range(np.size(item_list, 2)):
        minimum = np.min(item_list[:, :, channel])
        maximum = np.max(item_list[:, :, channel])
        if minimum < -360 or maximum > 360:
            print(channel, minimum, maximum)
            suspicious_channels.append(channel)
    for susp in suspicious_channels:
        print(BVH.get_joint_name(m, c, susp))
        for i in range(np.size(item_list, 0)):
            minimum = np.min(item_list[i, :, susp])
            maximum = np.max(item_list[i, :, susp])
            if minimum < -360 or maximum > 360:
                print(i, minimum, maximum, maximum - minimum)

# normalization (fuck the outliers)
# normalization to (-360, 360) -> (0-1)
_norm_scale = np.array([-360, 360])
item_list = data_prep.normalize(item_list, _norm_scale)

# data - label split
data_Y = item_list.copy()
data_X = item_list.copy()

for i, item in enumerate(item_list):
    item_bef = item[:surroundings[0]+1, :]
    item_aft = item[-surroundings[1]-1:, :]
    transition_length = np.size(item, 0) - surroundings[0] - surroundings[1]
    transition_approximation = data_prep.sign_synthesis(item_bef, item_aft, transition_length-1)
    new_item = np.concatenate((item_bef, transition_approximation, item_aft))
    data_X[i, :, :] = new_item
    # print(np.shape(item))
    # print(np.shape(new_item))
    # plt.plot(item[:, 20])
    # plt.plot(new_item[:, 20])
    # plt.show()

# shuffle

# train - test split
train_test_ratio = 0.2
